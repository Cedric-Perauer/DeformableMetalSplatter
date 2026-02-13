import Foundation

enum ModelIdentifier: Equatable, Hashable, Codable, CustomStringConvertible {
    case gaussianSplat(URL, useFP16: Bool, deformationEnabled: Bool = true)

    var description: String {
        switch self {
        case .gaussianSplat(let url, let useFP16, let deformationEnabled):
            "Gaussian Splat: \(url.path) (\(useFP16 ? "FP16" : "FP32")\(deformationEnabled ? "" : ", no deform"))"
        }
    }
    
    var useFP16: Bool {
        switch self {
        case .gaussianSplat(_, let useFP16, _):
            return useFP16
        }
    }
    
    var deformationEnabled: Bool {
        switch self {
        case .gaussianSplat(_, _, let deformationEnabled):
            return deformationEnabled
        }
    }
}
