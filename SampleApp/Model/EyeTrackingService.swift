#if os(iOS)

import ARKit
import Combine
import simd

/// Publishes smoothed head yaw/pitch deltas (in radians) relative to the pose captured on start,
/// sourced from ARKit face tracking (TrueDepth camera). Consumers can feed these into a scene
/// rotation controller to orbit the view in the same spirit as gyroscope-based "fake depth" mode.
class EyeTrackingService: NSObject, ObservableObject, ARSessionDelegate {

    // MARK: - Published state

    /// Smoothed head yaw delta in radians, relative to the reference pose (positive = head turns left
    /// in the device's native portrait frame).
    @Published var headYaw: Float = 0

    /// Smoothed head pitch delta in radians, relative to the reference pose (positive = head tilts up
    /// in the device's native portrait frame).
    @Published var headPitch: Float = 0

    /// Whether the service is actively running.
    @Published var isRunning: Bool = false

    /// Whether ARFaceTracking is supported on this device.
    static var isSupported: Bool {
        ARFaceTrackingConfiguration.isSupported
    }

    // MARK: - Configuration

    /// Smoothing factor (0 = no smoothing, 1 = frozen).
    var smoothingFactor: Float = 0.6

    // MARK: - Private

    private let session = ARSession()

    // Reference pose captured on start
    private var referenceYaw: Float?
    private var referencePitch: Float?

    // Smoothed radian deltas
    private var smoothedYaw: Float = 0
    private var smoothedPitch: Float = 0

    // MARK: - Lifecycle

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        guard EyeTrackingService.isSupported else {
            print("[HeadTracking] ARFaceTracking not supported on this device")
            return
        }
        referenceYaw = nil
        referencePitch = nil
        smoothedYaw = 0
        smoothedPitch = 0

        let config = ARFaceTrackingConfiguration()
        if ARFaceTrackingConfiguration.supportsWorldTracking {
            config.isWorldTrackingEnabled = false
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        print("[HeadTracking] Started face tracking session")
    }

    func stop() {
        session.pause()
        isRunning = false
        referenceYaw = nil
        referencePitch = nil
        DispatchQueue.main.async {
            self.headYaw = 0
            self.headPitch = 0
        }
        print("[HeadTracking] Stopped face tracking session")
    }

    /// Re-anchor the reference pose to the current head position.
    func recenter() {
        referenceYaw = nil
        referencePitch = nil
        smoothedYaw = 0
        smoothedPitch = 0
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let faceAnchor = anchors.compactMap({ $0 as? ARFaceAnchor }).first else { return }

        let transform = faceAnchor.transform
        let yaw = atan2(transform.columns.0.z, transform.columns.2.z)
        let pitch = asin(-transform.columns.1.z)

        if referenceYaw == nil {
            referenceYaw = yaw
            referencePitch = pitch
        }

        let deltaYaw = yaw - (referenceYaw ?? yaw)
        let deltaPitch = pitch - (referencePitch ?? pitch)

        smoothedYaw = smoothedYaw * smoothingFactor + deltaYaw * (1.0 - smoothingFactor)
        smoothedPitch = smoothedPitch * smoothingFactor + deltaPitch * (1.0 - smoothingFactor)

        let outYaw = smoothedYaw
        let outPitch = smoothedPitch
        DispatchQueue.main.async {
            self.headYaw = outYaw
            self.headPitch = outPitch
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[HeadTracking] Session failed: \(error.localizedDescription)")
        DispatchQueue.main.async {
            self.isRunning = false
        }
    }
}

#endif // os(iOS)
