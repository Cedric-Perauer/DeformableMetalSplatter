import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class DeformGraphSystem {
    let device: MTLDevice
    let mpsDevice: MPSGraphDevice
    let graph: MPSGraph
    var executable: MPSGraphExecutable?
    public let useFP16: Bool
    
    // Tensors
    var inputXYZTensor: MPSGraphTensor?
    var inputTTensor: MPSGraphTensor?
    var outXYZTensor: MPSGraphTensor?
    var outRotTensor: MPSGraphTensor?
    var outScaleTensor: MPSGraphTensor?
    
    var weightDataMap: [String: (Data, [NSNumber], MPSDataType)] = [:]
    
    // Deformation Network Params
    let W = 256
    let D = 8
    let SKIP_LAYER = 4
    let MULTIRES = 10
    let T_MULTIRES = 10
    
    // Larger batch size = fewer kernel launches = less overhead
    // 131072 should work on most devices; reduce if GPU crashes
    let SAFE_BATCH_SIZE = 16384
    var flatten = true
    public init(device: MTLDevice, useFP16: Bool) {
        self.device = device
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.graph = MPSGraph()
        #if arch(x86_64)
        self.useFP16 = false
        #else
        self.useFP16 = useFP16
        #endif
    }
    
    private static func float32DataToFloat16Data(_ data: Data) -> Data {
#if arch(x86_64)
        // Float16 isn't available on x86_64 (we alias it to Float elsewhere for buildability),
        // and FP16 deformation is disabled on that architecture anyway.
        return Data()
#else
        let floatCount = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { raw in
            let floats = raw.bindMemory(to: Float.self)
            var halfBits = [UInt16](repeating: 0, count: floatCount)
            for i in 0..<floatCount {
                halfBits[i] = Float16(floats[i]).bitPattern
            }
            return Data(bytes: halfBits, count: halfBits.count * MemoryLayout<UInt16>.size)
        }
#endif
    }
    
    public func loadWeights(flatData: Data) {
        var offset = 0
        let floatSize = MemoryLayout<Float>.size
        let totalCh = 63 + 21
        
        func extract(outDim: Int, inDim: Int, name: String) {
            let byteCount = outDim * inDim * floatSize
            guard (offset + byteCount) <= flatData.count else { return }
            
            let subDataF32 = flatData.subdata(in: offset..<(offset + byteCount))
            if useFP16 {
                let subDataF16 = Self.float32DataToFloat16Data(subDataF32)
                weightDataMap["\(name)_w"] = (subDataF16, [NSNumber(value: outDim), NSNumber(value: inDim)], .float16)
            } else {
                weightDataMap["\(name)_w"] = (subDataF32, [NSNumber(value: outDim), NSNumber(value: inDim)], .float32)
            }
            offset += byteCount
            
            let biasBytes = outDim * floatSize
            let biasDataF32 = flatData.subdata(in: offset..<(offset + biasBytes))
            if useFP16 {
                let biasDataF16 = Self.float32DataToFloat16Data(biasDataF32)
                weightDataMap["\(name)_b"] = (biasDataF16, [NSNumber(value: 1), NSNumber(value: outDim)], .float16)
            } else {
                weightDataMap["\(name)_b"] = (biasDataF32, [NSNumber(value: 1), NSNumber(value: outDim)], .float32)
            }
            offset += biasBytes
        }
        
        extract(outDim: W, inDim: totalCh, name: "L0")
        for i in 0..<(D-1) {
            let inDim = (i == SKIP_LAYER) ? (W + totalCh) : W
            extract(outDim: W, inDim: inDim, name: "L\(i+1)")
        }
        extract(outDim: 3, inDim: W, name: "Head_XYZ")
        extract(outDim: 4, inDim: W, name: "Head_Rot")
        extract(outDim: 3, inDim: W, name: "Head_Scale")
    }
    
    public func buildAndCompile() {
        let batchDim = NSNumber(value: -1) // Dynamic size
        let xyz: MPSGraphTensor
        let t: MPSGraphTensor
        
        // For some reason, I have to flatten the tensors otherwise the output doesn't match with the output generated from PyTorch :(
        if (self.flatten) {
            self.inputXYZTensor = graph.placeholder(shape: [batchDim], dataType: .float32, name: "in_xyz_flat")
            self.inputTTensor   = graph.placeholder(shape: [batchDim], dataType: .float32, name: "in_t_flat")
            guard let xyzFlat = self.inputXYZTensor, let tFlat = self.inputTTensor else { return }
            xyz = graph.reshape(xyzFlat, shape: [NSNumber(value: -1), NSNumber(value: 3)], name: "reshape_xyz")
            t   = graph.reshape(tFlat,   shape: [NSNumber(value: -1), NSNumber(value: 1)], name: "reshape_t")
        }
        else {
            self.inputXYZTensor = graph.placeholder(shape: [batchDim, 3], dataType: .float32, name: "in_xyz")
            self.inputTTensor   = graph.placeholder(shape: [batchDim, 1], dataType: .float32, name: "in_t")
            guard let inputXYZ = self.inputXYZTensor, let inputT = self.inputTTensor else { return }
            xyz = inputXYZ
            t   = inputT
        }
        
        let xyzTyped: MPSGraphTensor
        let tTyped: MPSGraphTensor
        if useFP16 {
            xyzTyped = graph.cast(xyz, to: .float16, name: "cast_xyz_f16")
            tTyped = graph.cast(t, to: .float16, name: "cast_t_f16")
        } else {
            xyzTyped = xyz
            tTyped = t
        }

        let embXYZ = positionalEncoding(input: xyzTyped, numFreqs: MULTIRES, dataType: useFP16 ? .float16 : .float32)
        let embT   = positionalEncoding(input: tTyped, numFreqs: T_MULTIRES, dataType: useFP16 ? .float16 : .float32)
        let inputs = graph.concatTensors([embXYZ, embT], dimension: 1, name: "input_concat")
        
        var h = inputs
        h = denseLayer(input: h, name: "L0", activation: true)
        
        for i in 0..<(D-1) {
            if i == SKIP_LAYER { h = graph.concatTensors([inputs, h], dimension: 1, name: "skip") }
            h = denseLayer(input: h, name: "L\(i+1)", activation: true)
        }
        
        // Output Layers
        let outXYZRaw = denseLayer(input: h, name: "Head_XYZ", activation: false)
        let outRotRaw = denseLayer(input: h, name: "Head_Rot", activation: false)
        let outScaleRaw = denseLayer(input: h, name: "Head_Scale", activation: false)
        
        // We always write float32 into the output MTLBuffers.
        let outXYZ = useFP16 ? graph.cast(outXYZRaw, to: .float32, name: "out_xyz_f32") : outXYZRaw
        let outRot = useFP16 ? graph.cast(outRotRaw, to: .float32, name: "out_rot_f32") : outRotRaw
        let outScale = useFP16 ? graph.cast(outScaleRaw, to: .float32, name: "out_scale_f32") : outScaleRaw
        
        let feedsDict: [MPSGraphTensor : MPSGraphShapedType]
        if (self.flatten) {
            self.outXYZTensor = graph.reshape(outXYZ, shape: [batchDim], name: "out_xyz_flat")
            self.outRotTensor = graph.reshape(outRot, shape: [batchDim], name: "out_rot_flat")
            self.outScaleTensor = graph.reshape(outScale, shape: [batchDim], name: "out_scale_flat")
            feedsDict = [
                inputXYZTensor!: MPSGraphShapedType(shape: [batchDim], dataType: .float32),
                inputTTensor!:   MPSGraphShapedType(shape: [batchDim], dataType: .float32)
            ]
        }
        else {
            self.outXYZTensor   = outXYZ
            self.outRotTensor   = outRot
            self.outScaleTensor = outScale
            feedsDict = [
                inputXYZTensor!: MPSGraphShapedType(shape: [batchDim, 3], dataType: .float32),
                inputTTensor!:   MPSGraphShapedType(shape: [batchDim, 1], dataType: .float32)
            ]
        }
        
        self.executable = graph.compile(with: mpsDevice,
                                        feeds: feedsDict,
                                        targetTensors: [outXYZTensor!, outRotTensor!, outScaleTensor!],
                                        targetOperations: nil,
                                        compilationDescriptor: nil)
    }
    
    // Helper function to slice the buffer with byte offset
    private func createTensorView(buffer: MTLBuffer,
                                  offset: Int,
                                  shape: [NSNumber]) -> MPSGraphTensorData {
        
        let desc = MPSNDArrayDescriptor(dataType: .float32, shape: shape)
        let ndArray = MPSNDArray(buffer: buffer, offset: offset, descriptor: desc)
        return MPSGraphTensorData(ndArray)
    }
    
    private var didPrintDebug = false
    
    public func run(commandQueue: MTLCommandQueue,
                    xyzBuffer: MTLBuffer,
                    tBuffer: MTLBuffer,
                    outXYZ: MTLBuffer,
                    outRot: MTLBuffer,
                    outScale: MTLBuffer,
                    count: Int) {
        
        guard let exec = executable else { return }
        let totalStart = CFAbsoluteTimeGetCurrent()
        let floatSize = MemoryLayout<Float>.size
        let numBatches = (count + SAFE_BATCH_SIZE - 1) / SAFE_BATCH_SIZE
        
        // Debug: Print feed and target tensor info once
        if !didPrintDebug {
            didPrintDebug = true
            print("=== DeformGraphSystem Debug ===")
            print("useFP16: \(useFP16), flatten: \(flatten)")
            if let feedTensors = exec.feedTensors {
                print("Feed tensors (\(feedTensors.count)):")
                for (i, tensor) in feedTensors.enumerated() {
                    print("  [\(i)] op='\(tensor.operation.name ?? "nil")', shape=\(tensor.shape ?? [])")
                }
            }
            if let targetTensors = exec.targetTensors {
                print("Target tensors (\(targetTensors.count)):")
                for (i, tensor) in targetTensors.enumerated() {
                    print("  [\(i)] op='\(tensor.operation.name ?? "nil")', shape=\(tensor.shape ?? [])")
                }
            }
            print("================================")
        }

        for i in stride(from: 0, to: count, by: SAFE_BATCH_SIZE) {
            autoreleasepool {
                let batchStart = CFAbsoluteTimeGetCurrent()
                let currentCount = min(SAFE_BATCH_SIZE, count - i)
                
                // Define the offset
                let offsetXYZ = i * 3 * floatSize
                let offsetT   = i * 1 * floatSize
                let offsetOutXYZ = i * 3 * floatSize
                let offsetOutRot = i * 4 * floatSize
                let offsetOutScale = i * 3 * floatSize
                
                let xyzShape: [NSNumber]
                let tShape: [NSNumber]
                let outRotShape: [NSNumber]
                let outScaleShape: [NSNumber]
                
                // Get the size of the batched buffer
                if (self.flatten) {
                    xyzShape = [NSNumber(value: currentCount * 3)]
                    tShape = [NSNumber(value: currentCount * 1)]
                    outRotShape = [NSNumber(value: currentCount * 4)]
                    outScaleShape = [NSNumber(value: currentCount * 3)]
                }
                else {
                    xyzShape   = [NSNumber(value: currentCount), 3]
                    tShape     = [NSNumber(value: currentCount), 1]
                    outRotShape   = [NSNumber(value: currentCount), 4]
                    outScaleShape = [NSNumber(value: currentCount), 3]
                }
                
                // Get the batched buffer
                let xyzData = createTensorView(buffer: xyzBuffer, offset: offsetXYZ, shape: xyzShape)
                let tData = createTensorView(buffer: tBuffer, offset: offsetT, shape: tShape)
                
                let outXYZData = createTensorView(buffer: outXYZ, offset: offsetOutXYZ, shape: xyzShape)
                let outRotData = createTensorView(buffer: outRot, offset: offsetOutRot, shape: outRotShape)
                let outScaleData = createTensorView(buffer: outScale, offset: offsetOutScale, shape: outScaleShape)
                
                var inputsArray: [MPSGraphTensorData] = []
                var resultsArray: [MPSGraphTensorData] = []
                
                if let feedTensors = exec.feedTensors {
                    for tensor in feedTensors {
                        let opName = tensor.operation.name ?? ""
                        // Match by operation name - check for xyz or t in the name
                        if opName.contains("xyz") { 
                            inputsArray.append(xyzData) 
                        }
                        else if opName.contains("t") && !opName.contains("xyz") { 
                            inputsArray.append(tData) 
                        }
                        // Fallback: Check shape size for non-flatten mode
                        else if !self.flatten, let shape = tensor.shape, shape.count > 1 {
                            if shape[1].intValue == 3 { inputsArray.append(xyzData) }
                            else { inputsArray.append(tData) }
                        }
                        // Fallback for flatten mode: check 1D shape
                        else if self.flatten, let shape = tensor.shape, shape.count == 1 {
                            // xyz has 3x the count of t
                            let size = shape[0].intValue
                            if size == currentCount * 3 { inputsArray.append(xyzData) }
                            else { inputsArray.append(tData) }
                        }
                        else {
                            // Last resort - assume first is xyz, second is t based on order
                            print("WARNING: Could not determine tensor type for '\(opName)', shape: \(tensor.shape ?? [])")
                            inputsArray.append(inputsArray.isEmpty ? xyzData : tData)
                        }
                    }
                }

                resultsArray.append(outXYZData)
                resultsArray.append(outRotData)
                resultsArray.append(outScaleData)
                
                if inputsArray.count == (exec.feedTensors?.count ?? 0) {
                    let _ = exec.run(with: commandQueue,
                                     inputs: inputsArray,
                                     results: resultsArray,
                                     executionDescriptor: nil)
                }
                
                let batchElapsedMs = (CFAbsoluteTimeGetCurrent() - batchStart) * 1000.0
                if numBatches > 1 {
                    print("DeformGraph batch \(i / SAFE_BATCH_SIZE): \(batchElapsedMs) ms")
                }
            }
        }

        let totalElapsedMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000.0
        print("DeformGraph total (\(numBatches) batches): \(totalElapsedMs) ms")
    }
    
    func positionalEncoding(input: MPSGraphTensor, numFreqs: Int, dataType: MPSDataType) -> MPSGraphTensor {
        var tensors = [input]
        for i in 0..<numFreqs {
            let freq = Float(pow(2.0, Double(i)))
            let freqConst = graph.constant(Double(freq), dataType: dataType)
            let scaled = graph.multiplication(input, freqConst, name: nil)
            tensors.append(graph.sin(with: scaled, name: nil))
            tensors.append(graph.cos(with: scaled, name: nil))
        }
        return graph.concatTensors(tensors, dimension: 1, name: nil)
    }
    
    func denseLayer(input: MPSGraphTensor, name: String, activation: Bool) -> MPSGraphTensor {
        guard let (wD, wS, wType) = weightDataMap["\(name)_w"],
              let (bD, bS, bType) = weightDataMap["\(name)_b"] else {
            print("CRITICAL ERROR: Missing weights for layer '\(name)'. Loaded keys: \(weightDataMap.keys)")
            fatalError("Missing weights for \(name)")
        }
        let w = graph.constant(wD, shape: wS, dataType: wType)
        let b = graph.constant(bD, shape: bS, dataType: bType)
        let wT = graph.transposeTensor(w, dimension: 0, withDimension: 1, name: nil)
        var out = graph.matrixMultiplication(primary: input, secondary: wT, name: nil)
        out = graph.addition(out, b, name: nil)
        if activation { out = graph.reLU(with: out, name: nil) }
        return out
    }
    
    // MARK: - Test Vector Comparison
    
    /// Test vector structure matching the Python-generated test_vectors.bin format
    public struct TestVector {
        public var xyz: SIMD3<Float>
        public var t: Float
        public var expectedDXYZ: SIMD3<Float>
        public var expectedDRot: SIMD4<Float>
        public var expectedDScale: SIMD3<Float>
    }
    
    /// Load test vectors from a binary file
    public static func loadTestVectors(from url: URL) -> [TestVector]? {
        guard let data = try? Data(contentsOf: url) else {
            print("Failed to read test vectors from \(url)")
            return nil
        }
        
        let floatSize = MemoryLayout<Float>.size
        var offset = 0
        
        // Read count (uint32)
        guard data.count >= 4 else { return nil }
        let count: UInt32 = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
        offset += 4
        
        // Each sample: 3 (xyz) + 1 (t) + 3 (d_xyz) + 4 (d_rot) + 3 (d_scale) = 14 floats
        let sampleSize = 14 * floatSize
        guard data.count >= 4 + Int(count) * sampleSize else {
            print("Test vectors file too small: expected \(4 + Int(count) * sampleSize), got \(data.count)")
            return nil
        }
        
        var vectors: [TestVector] = []
        
        for _ in 0..<count {
            let xyz = SIMD3<Float>(
                data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + floatSize, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 2 * floatSize, as: Float.self) }
            )
            offset += 3 * floatSize
            
            let t = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
            offset += floatSize
            
            let dXYZ = SIMD3<Float>(
                data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + floatSize, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 2 * floatSize, as: Float.self) }
            )
            offset += 3 * floatSize
            
            let dRot = SIMD4<Float>(
                data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + floatSize, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 2 * floatSize, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 3 * floatSize, as: Float.self) }
            )
            offset += 4 * floatSize
            
            let dScale = SIMD3<Float>(
                data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + floatSize, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 2 * floatSize, as: Float.self) }
            )
            offset += 3 * floatSize
            
            vectors.append(TestVector(
                xyz: xyz,
                t: t,
                expectedDXYZ: dXYZ,
                expectedDRot: dRot,
                expectedDScale: dScale
            ))
        }
        
        print("Loaded \(vectors.count) test vectors")
        return vectors
    }
    
    /// Run MPS graph on test vectors and compare with expected outputs
    public func runTestComparison(testVectors: [TestVector], commandQueue: MTLCommandQueue) {
        guard !testVectors.isEmpty else {
            print("No test vectors to compare")
            return
        }
        
        let count = testVectors.count
        let floatSize = MemoryLayout<Float>.size
        
        // Allocate buffers
        guard let xyzBuffer = device.makeBuffer(length: count * 3 * floatSize, options: .storageModeShared),
              let tBuffer = device.makeBuffer(length: count * floatSize, options: .storageModeShared),
              let outXYZBuffer = device.makeBuffer(length: count * 3 * floatSize, options: .storageModeShared),
              let outRotBuffer = device.makeBuffer(length: count * 4 * floatSize, options: .storageModeShared),
              let outScaleBuffer = device.makeBuffer(length: count * 3 * floatSize, options: .storageModeShared) else {
            print("Failed to allocate test buffers")
            return
        }
        
        // Fill input buffers
        let xyzPtr = xyzBuffer.contents().bindMemory(to: Float.self, capacity: count * 3)
        let tPtr = tBuffer.contents().bindMemory(to: Float.self, capacity: count)
        
        for (i, vec) in testVectors.enumerated() {
            xyzPtr[i * 3 + 0] = vec.xyz.x
            xyzPtr[i * 3 + 1] = vec.xyz.y
            xyzPtr[i * 3 + 2] = vec.xyz.z
            tPtr[i] = vec.t
        }
        
        // Run MPS graph
        print("\n=== Running MPS Graph Comparison Test ===")
        print("Testing \(count) vectors...")
        
        run(commandQueue: commandQueue,
            xyzBuffer: xyzBuffer,
            tBuffer: tBuffer,
            outXYZ: outXYZBuffer,
            outRot: outRotBuffer,
            outScale: outScaleBuffer,
            count: count)
        
        // Read outputs
        let outXYZPtr = outXYZBuffer.contents().bindMemory(to: Float.self, capacity: count * 3)
        let outRotPtr = outRotBuffer.contents().bindMemory(to: Float.self, capacity: count * 4)
        let outScalePtr = outScaleBuffer.contents().bindMemory(to: Float.self, capacity: count * 3)
        
        // Compare
        var maxDiffXYZ: Float = 0
        var maxDiffRot: Float = 0
        var maxDiffScale: Float = 0
        var totalDiffXYZ: Float = 0
        var totalDiffRot: Float = 0
        var totalDiffScale: Float = 0
        
        print("\nFirst 10 comparisons:")
        print(String(repeating: "-", count: 120))
        
        for (i, vec) in testVectors.enumerated() {
            let mpsXYZ = SIMD3<Float>(outXYZPtr[i*3+0], outXYZPtr[i*3+1], outXYZPtr[i*3+2])
            let mpsRot = SIMD4<Float>(outRotPtr[i*4+0], outRotPtr[i*4+1], outRotPtr[i*4+2], outRotPtr[i*4+3])
            let mpsScale = SIMD3<Float>(outScalePtr[i*3+0], outScalePtr[i*3+1], outScalePtr[i*3+2])
            
            let diffXYZ = abs(mpsXYZ - vec.expectedDXYZ)
            let diffRot = abs(mpsRot - vec.expectedDRot)
            let diffScale = abs(mpsScale - vec.expectedDScale)
            
            let maxDX = max(diffXYZ.x, max(diffXYZ.y, diffXYZ.z))
            let maxDR = max(diffRot.x, max(diffRot.y, max(diffRot.z, diffRot.w)))
            let maxDS = max(diffScale.x, max(diffScale.y, diffScale.z))
            
            maxDiffXYZ = max(maxDiffXYZ, maxDX)
            maxDiffRot = max(maxDiffRot, maxDR)
            maxDiffScale = max(maxDiffScale, maxDS)
            
            totalDiffXYZ += diffXYZ.x + diffXYZ.y + diffXYZ.z
            totalDiffRot += diffRot.x + diffRot.y + diffRot.z + diffRot.w
            totalDiffScale += diffScale.x + diffScale.y + diffScale.z
            
            if i < 10 {
                print("[\(i)] xyz=(\(String(format: "%.4f", vec.xyz.x)), \(String(format: "%.4f", vec.xyz.y)), \(String(format: "%.4f", vec.xyz.z))), t=\(String(format: "%.4f", vec.t))")
                print("     Expected d_xyz: (\(String(format: "%.6f", vec.expectedDXYZ.x)), \(String(format: "%.6f", vec.expectedDXYZ.y)), \(String(format: "%.6f", vec.expectedDXYZ.z)))")
                print("     MPS      d_xyz: (\(String(format: "%.6f", mpsXYZ.x)), \(String(format: "%.6f", mpsXYZ.y)), \(String(format: "%.6f", mpsXYZ.z)))")
                print("     Diff:           (\(String(format: "%.2e", diffXYZ.x)), \(String(format: "%.2e", diffXYZ.y)), \(String(format: "%.2e", diffXYZ.z)))")
                print("     Expected d_rot: (\(String(format: "%.6f", vec.expectedDRot.x)), \(String(format: "%.6f", vec.expectedDRot.y)), \(String(format: "%.6f", vec.expectedDRot.z)), \(String(format: "%.6f", vec.expectedDRot.w)))")
                print("     MPS      d_rot: (\(String(format: "%.6f", mpsRot.x)), \(String(format: "%.6f", mpsRot.y)), \(String(format: "%.6f", mpsRot.z)), \(String(format: "%.6f", mpsRot.w)))")
                print("")
            }
        }
        
        print(String(repeating: "=", count: 120))
        print("SUMMARY:")
        print("  Max diff d_xyz:   \(String(format: "%.2e", maxDiffXYZ))")
        print("  Max diff d_rot:   \(String(format: "%.2e", maxDiffRot))")
        print("  Max diff d_scale: \(String(format: "%.2e", maxDiffScale))")
        print("  Avg diff d_xyz:   \(String(format: "%.2e", totalDiffXYZ / Float(count * 3)))")
        print("  Avg diff d_rot:   \(String(format: "%.2e", totalDiffRot / Float(count * 4)))")
        print("  Avg diff d_scale: \(String(format: "%.2e", totalDiffScale / Float(count * 3)))")
        
        let tolerance: Float = 1e-4
        if maxDiffXYZ < tolerance && maxDiffRot < tolerance && maxDiffScale < tolerance {
            print("\n✓ PASS: MPS output matches PyTorch within tolerance (\(tolerance))")
        } else {
            print("\n✗ FAIL: MPS output differs from PyTorch!")
            print("  This indicates a bug in the MPS graph implementation.")
        }
        print(String(repeating: "=", count: 120))
    }
}
