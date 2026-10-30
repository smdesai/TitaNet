// SpeakerEngine — samples in, a 192-d TitaNet embedding out, on either runtime: the Core AI bundle (titanet.aimodel)
// or the Core ML export (fp32 / fp16c), from the folder ModelStore downloaded it to. Owns the model and the mel front
// end (whose two small constant files stay in the app bundle), off the main actor. Every
// load and embed runs one at a time, in the order asked: a model is not called concurrently, and an embed asked for
// while the model is still loading waits for it.

import CoreAI
import CoreML
import Foundation

enum Runtime: String, CaseIterable, Identifiable, Sendable {
    case coreAI = "coreai", coreMLFP32 = "coreml-fp32", coreMLFP16 = "coreml-fp16"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .coreAI: return "Core AI"
        case .coreMLFP32: return "Core ML fp32"
        case .coreMLFP16: return "Core ML fp16"
        }
    }

    var isCoreML: Bool { self != .coreAI }
}

enum ComputeUnit: String, CaseIterable, Identifiable, Sendable {
    case auto = "default", gpu, ane, cpu

    var id: String { rawValue }

    var label: String {
        switch self {
        case .auto: return "Auto"
        case .gpu: return "GPU"
        case .ane: return "Neural Engine"
        case .cpu: return "CPU"
        }
    }

    var short: String { self == .ane ? "ANE" : label }

    /// Core AI: tnl/runtime.options
    var options: SpecializationOptions {
        switch self {
        case .cpu: return .cpuOnly
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .ane: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .auto: return .default
        }
    }

    /// Core ML: the CPU is always allowed as a fallback, as Core ML requires.
    var mlUnits: MLComputeUnits {
        switch self {
        case .cpu: return .cpuOnly
        case .gpu: return .cpuAndGPU
        case .ane: return .cpuAndNeuralEngine
        case .auto: return .all
        }
    }
}

struct EngineInfo: Sendable, Equatable {
    var runtime: Runtime
    var recipe: String              // which export: w_scan_f512 · wfp16, c_scan_f512 · fp32, ...
    var window: Int                 // F, mel frames per scan call
    var windowSeconds: Double
    var margin: Int
    var mels: Int
    var entrypoints: [String]
    var architecture: String
    var compute: ComputeUnit
    var loadSeconds: Double
}

struct Embedding: Sendable, Codable, Equatable {
    var vector: [Float]
    var frames: Int                 // valid mel frames (10 ms each)
    var windows: Int
    var calls: Int
    var featureSeconds: Double      // the mel front end
    var modelSeconds: Double        // scan sweeps + head
    var engine: String = ""         // the runtime and compute unit that produced it, e.g. "Core ML fp16 · ANE"
}

actor SpeakerEngine {
    static let sampleRate = 16000

    private var model: (any WindowedModel)?
    private var engineLabel = ""
    private var mel: MelFrontend?
    private var tail: Task<Void, Never>?

    /// `folder`: where ModelStore put the runtime's files.
    func load(_ runtime: Runtime, _ unit: ComputeUnit, from folder: URL) async throws -> EngineInfo {
        try await serially { try await self.loadNow(runtime, unit, folder) }
    }

    func embed(_ samples: [Float], progress: @MainActor @Sendable @escaping (String) -> Void) async throws -> Embedding {
        try await serially { try await self.embedNow(samples, progress: progress) }
    }

    private func serially<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) async throws -> T {
        let previous = tail
        let job = Task { await previous?.value; return try await body() }
        tail = Task { _ = try? await job.value }
        return try await job.value
    }

    private func loadNow(_ runtime: Runtime, _ unit: ComputeUnit, _ folder: URL) async throws -> EngineInfo {
        model = nil     // release the old model before building the next
        if mel == nil {
            let fb = try Self.floats("mel_fb", count: MelFrontend.nMels * MelFrontend.bins)
            let w = try Self.floats("mel_window", count: MelFrontend.win)
            mel = MelFrontend(filterbank: fb, window: w)
        }
        let t0 = ContinuousClock.now
        let m: any WindowedModel
        let recipe: String
        let entrypoints: [String]
        if runtime.isCoreML {
            let c = try await CoreMLModel(folder: folder, units: unit.mlUnits)
            m = c
            recipe = "c_scan_f512 · \(c.record.precision) ML Program"
            entrypoints = c.record.models.keys.sorted { $0 > $1 }
        } else {
            let graph = folder.appendingPathComponent("titanet.aimodel")
            let exportURL = folder.appendingPathComponent("export.json")
            let record = try JSONDecoder().decode(ExportRecord.self, from: Data(contentsOf: exportURL))
            m = try await CoreAIModel(graph: graph, record: record, options: unit.options)
            recipe = "w_scan_f512 · \(record.arm ?? "—")"
            entrypoints = record.entrypoints
        }
        guard m.nMels == MelFrontend.nMels else {
            throw StudioError.contract("the model takes \(m.nMels) mels, the front end makes \(MelFrontend.nMels)")
        }
        // One throwaway call of each function: Core ML on the GPU builds its pipelines on the first prediction
        // (measured on iPhone19,4: a 4 s clip took 1.2-2.0 s cold, 0.1 s warm), so pay that here, not on the first clip.
        let zeros = [Float](repeating: 0, count: m.cOut)
        _ = try await m.scan(mel: [Float](repeating: 0, count: m.nMels * m.F), valid: [Float](repeating: 1, count: m.F),
                             keep: [Float](repeating: 1, count: m.F), nInv: 1 / Float(m.F),
                             sums: m.seSizes.map { [Float](repeating: 0, count: $0) }, mean: zeros,
                             std: [Float](repeating: 1, count: m.cOut))
        _ = try await m.head(mu: zeros, sg: [Float](repeating: 1, count: m.cOut))
        model = m
        engineLabel = "\(runtime.label) · \(unit.short)"
        return EngineInfo(runtime: runtime, recipe: recipe, window: m.F,
                          windowSeconds: Double(m.F * MelFrontend.hop) / Double(Self.sampleRate), margin: m.margin,
                          mels: m.nMels, entrypoints: entrypoints, architecture: AIModel.deviceArchitectureName,
                          compute: unit, loadSeconds: seconds(since: t0))
    }

    private func embedNow(_ samples: [Float], progress: @MainActor @Sendable @escaping (String) -> Void) async throws -> Embedding {
        guard let model, let mel else { throw StudioError.contract("the model is not loaded") }
        await progress("Extracting log-mel features")
        let t0 = ContinuousClock.now
        let (m, frames, n) = mel(samples)
        guard n >= 1 else { throw StudioError.audio("The clip is too short to embed.") }
        // repack the n valid frames into a tight [80, n] buffer: the window slicer's row stride is n
        var packed = [Float](repeating: 0, count: MelFrontend.nMels * n)
        for c in 0..<MelFrontend.nMels {
            let src = c * frames, dst = c * n
            for t in 0..<n { packed[dst + t] = m[src + t] }
        }
        let featureSeconds = seconds(since: t0)
        let t1 = ContinuousClock.now
        let r = try await windowedEmbed(model, mel: packed, n: n, progress: progress)
        let vector = r.emb.map { Float($0) }
        guard vector.allSatisfy(\.isFinite) else { throw StudioError.contract("the embedding is not finite") }
        return Embedding(vector: vector, frames: n, windows: r.windows, calls: r.calls,
                         featureSeconds: featureSeconds, modelSeconds: seconds(since: t1), engine: engineLabel)
    }

    private static func floats(_ name: String, count: Int) throws -> [Float] {
        guard let url = Bundle.main.url(forResource: name, withExtension: "f32") else {
            throw StudioError.contract("\(name).f32 is missing from the app bundle")
        }
        let d = try Data(contentsOf: url)
        guard d.count == 4 * count else { throw StudioError.contract("\(name).f32: \(d.count) bytes, expected \(4 * count)") }
        return d.withUnsafeBytes { raw in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self))) }
        }
    }
}

/// NeMo's speaker-verification score (verify_speakers): (cos + 1) / 2 on the raw embeddings, same speaker at >= 0.7.
enum Scoring {
    static let threshold = 0.7

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var ab = 0.0, aa = 0.0, bb = 0.0
        for i in 0..<min(a.count, b.count) {
            let x = Double(a[i]), y = Double(b[i])
            ab += x * y
            aa += x * x
            bb += y * y
        }
        let d = (aa * bb).squareRoot()
        return d == 0 ? 0 : ab / d
    }

    static func score(_ a: [Float], _ b: [Float]) -> Double { (cosine(a, b) + 1) / 2 }
}

func seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}
