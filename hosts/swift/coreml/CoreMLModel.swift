// CoreMLModel — the Core ML export behind the WindowedModel interface: TitaNetScan_<variant> and TitaNetHead_<variant>,
// as Xcode compiles the .mlpackage files into the app bundle, described by export_<variant>.json.

import CoreML
import Foundation

/// export_<variant>.json: the window, its margin, the two models and (fp16c) the attention moments' scale.
struct CoreMLExportRecord: Decodable, Sendable {
    struct Model: Decodable, Sendable { var file: String; var inputs: [String]; var outputs: [String] }
    var precision: String
    var shape: Int
    var margin: Int
    var models: [String: Model]
    var moment_scale: Double?       // fp16c: the attention moments come back centred and scaled by this
}

final class CoreMLModel: WindowedModel {
    let record: CoreMLExportRecord
    let F: Int
    let margin: Int
    let nMels = 80
    let cOut = 3072
    let seSizes = [1024, 1024, 1024, 1024, 3072]
    let momentScale: Double?
    private let scanModel: MLModel
    private let headModel: MLModel

    /// `record`: export_<variant>.json. `compiled(name)`: where the compiled model of that .mlpackage file lives, e.g.
    /// Bundle.main.url(forResource: name, withExtension: "mlmodelc") for models added to an Xcode target.
    init(record rec: CoreMLExportRecord, units: MLComputeUnits, compiled: (String) -> URL?) async throws {
        guard let s = rec.models["scan"], let h = rec.models["head"] else {
            throw TitaNetError.contract("export json: no scan / head model")
        }
        func url(_ file: String) throws -> URL {
            let name = (file as NSString).deletingPathExtension
            guard let u = compiled(name) else { throw TitaNetError.contract("\(name).mlmodelc is missing") }
            return u
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        scanModel = try await MLModel.load(contentsOf: try url(s.file), configuration: cfg)
        headModel = try await MLModel.load(contentsOf: try url(h.file), configuration: cfg)
        record = rec
        F = rec.shape
        margin = rec.margin
        momentScale = rec.moment_scale
    }

    func scan(mel: [Float], valid: [Float], keep: [Float], nInv: Float, sums: [[Float]], mean: [Float], std: [Float])
        async throws -> [String: [Double]] {
        var d: [String: MLFeatureValue] = [
            "mel": MLFeatureValue(multiArray: try Self.array(mel, [1, nMels, F])),
            "valid": MLFeatureValue(multiArray: try Self.array(valid, [1, 1, F])),
            "keep": MLFeatureValue(multiArray: try Self.array(keep, [1, 1, F])),
            "n_inv": MLFeatureValue(multiArray: try Self.array([nInv], [1, 1])),
            "mean": MLFeatureValue(multiArray: try Self.array(mean, [1, cOut])),
            "std": MLFeatureValue(multiArray: try Self.array(std, [1, cOut])),
        ]
        for j in 0..<5 { d["s\(j)"] = MLFeatureValue(multiArray: try Self.array(sums[j], [1, sums[j].count])) }
        nonisolated(unsafe) let model = scanModel      // MLModel predictions are thread-safe (Core ML docs)
        let o = try await model.prediction(from: try MLDictionaryFeatureProvider(dictionary: d))
        var r: [String: [Double]] = [:]
        for name in o.featureNames {
            if let a = o.featureValue(for: name)?.multiArrayValue { r[name] = Self.doubles(a) }
        }
        return r
    }

    func head(mu: [Float], sg: [Float]) async throws -> [Double] {
        let feats = try MLDictionaryFeatureProvider(dictionary: [
            "mu": MLFeatureValue(multiArray: try Self.array(mu, [1, cOut])),
            "sg": MLFeatureValue(multiArray: try Self.array(sg, [1, cOut])),
        ])
        nonisolated(unsafe) let model = headModel
        let o = try await model.prediction(from: feats)
        guard let e = o.featureValue(for: "emb")?.multiArrayValue else { throw TitaNetError.contract("head: no emb output") }
        return Self.doubles(e)
    }

    static func array(_ v: [Float], _ shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        a.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, strides in
            // freshly allocated arrays may pad the last axis: honor the strides
            let last = shape.count - 1
            let rows = shape.dropLast().reduce(1, *)
            let width = shape[last]
            let rowStride = last > 0 ? strides[last - 1] : width
            for r in 0..<rows {
                for c in 0..<width { ptr[r * rowStride + c * strides[last]] = v[r * width + c] }
            }
        }
        return a
    }

    /// Every output here is [1, C]; read it as Double whatever its storage type.
    static func doubles(_ a: MLMultiArray) -> [Double] {
        let n = a.shape.last?.intValue ?? a.count
        let s = a.strides.last?.intValue ?? 1
        switch a.dataType {
        case .float16:
            // never for these models (every output is declared fp32), but kept correct on macOS 14 too
            if #available(macOS 15.0, *) {
                return a.withUnsafeBufferPointer(ofType: Float16.self) { p in (0..<n).map { Double(p[$0 * s]) } }
            }
            return (0..<n).map { a[$0 * s].doubleValue }
        case .double:
            return a.withUnsafeBufferPointer(ofType: Double.self) { p in (0..<n).map { p[$0 * s] } }
        default:
            return a.withUnsafeBufferPointer(ofType: Float.self) { p in (0..<n).map { Double(p[$0 * s]) } }
        }
    }
}
