// SpeakerEngine — 16 kHz mono samples in, the 192-d TitaNet embedding out, on Core ML. Owns the models and the mel
// front end, off the main actor; every load and embed runs one at a time, in order. Expects in the app bundle:
// TitaNetScan_<variant>.mlpackage and TitaNetHead_<variant>.mlpackage (Xcode compiles them to .mlmodelc),
// export_<variant>.json, mel_fb.f32 and mel_window.f32.

import CoreML
import Foundation

enum Variant: String, CaseIterable, Sendable {
    case fp32       // CPU / GPU; matches NeMo to ~1e-9 in cosine
    case fp16c      // Neural Engine; worst cosine to NeMo 0.9968 on the ANE, 0.9997 on the GPU
}

enum ComputeUnit: String, CaseIterable, Sendable {
    case all, gpu, ane, cpu

    /// Core ML always allows the CPU as a fallback.
    var mlUnits: MLComputeUnits {
        switch self {
        case .cpu: return .cpuOnly
        case .gpu: return .cpuAndGPU
        case .ane: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }
}

struct Embedding: Sendable, Equatable {
    var vector: [Float]
    var frames: Int                 // valid mel frames (10 ms each)
    var windows: Int
    var calls: Int
    var seconds: Double             // mel front end + model
}

actor SpeakerEngine {
    static let sampleRate = 16000

    private var model: CoreMLModel?
    private var mel: MelFrontend?
    private var tail: Task<Void, Never>?

    /// Loads (or switches to) a variant. fp16c on `.ane` is the fast path; fp32 on `.gpu` the exact one.
    func load(_ variant: Variant, _ unit: ComputeUnit) async throws {
        try await serially { try await self.loadNow(variant, unit) }
    }

    func embed(_ samples: [Float], progress: @MainActor @Sendable @escaping (String) -> Void = { _ in }) async throws -> Embedding {
        try await serially { try await self.embedNow(samples, progress: progress) }
    }

    private func serially<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) async throws -> T {
        let previous = tail
        let job = Task { await previous?.value; return try await body() }
        tail = Task { _ = try? await job.value }
        return try await job.value
    }

    private func loadNow(_ variant: Variant, _ unit: ComputeUnit) async throws {
        model = nil
        if mel == nil {
            let fb = try Self.floats("mel_fb", count: MelFrontend.nMels * MelFrontend.bins)
            let w = try Self.floats("mel_window", count: MelFrontend.win)
            mel = MelFrontend(filterbank: fb, window: w)
        }
        guard let recURL = Bundle.main.url(forResource: "export_\(variant.rawValue)", withExtension: "json") else {
            throw TitaNetError.contract("export_\(variant.rawValue).json is missing from the app bundle")
        }
        let rec = try JSONDecoder().decode(CoreMLExportRecord.self, from: Data(contentsOf: recURL))
        let m = try await CoreMLModel(record: rec, units: unit.mlUnits) {
            Bundle.main.url(forResource: $0, withExtension: "mlmodelc")
        }
        // One throwaway call of each model: Core ML on the GPU builds its pipelines on the first prediction
        // (measured on an iPhone: a 4 s clip took 1.2-2.0 s cold, 0.1 s warm), so pay that here.
        let zeros = [Float](repeating: 0, count: m.cOut)
        _ = try await m.scan(mel: [Float](repeating: 0, count: m.nMels * m.F), valid: [Float](repeating: 1, count: m.F),
                             keep: [Float](repeating: 1, count: m.F), nInv: 1 / Float(m.F),
                             sums: m.seSizes.map { [Float](repeating: 0, count: $0) }, mean: zeros,
                             std: [Float](repeating: 1, count: m.cOut))
        _ = try await m.head(mu: zeros, sg: [Float](repeating: 1, count: m.cOut))
        model = m
    }

    private func embedNow(_ samples: [Float], progress: @MainActor @Sendable @escaping (String) -> Void) async throws -> Embedding {
        guard let model, let mel else { throw TitaNetError.contract("the model is not loaded") }
        let t0 = ContinuousClock.now
        let (m, frames, n) = mel(samples)
        guard n >= 1 else { throw TitaNetError.audio("The clip is too short to embed.") }
        // repack the n valid frames into a tight [80, n] buffer: the window slicer's row stride is n
        var packed = [Float](repeating: 0, count: MelFrontend.nMels * n)
        for c in 0..<MelFrontend.nMels {
            let src = c * frames, dst = c * n
            for t in 0..<n { packed[dst + t] = m[src + t] }
        }
        let r = try await windowedEmbed(model, mel: packed, n: n, progress: progress)
        let vector = r.emb.map { Float($0) }
        guard vector.allSatisfy(\.isFinite) else { throw TitaNetError.contract("the embedding is not finite") }
        let d = ContinuousClock.now - t0
        return Embedding(vector: vector, frames: n, windows: r.windows, calls: r.calls,
                         seconds: Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18)
    }

    private static func floats(_ name: String, count: Int) throws -> [Float] {
        guard let url = Bundle.main.url(forResource: name, withExtension: "f32") else {
            throw TitaNetError.contract("\(name).f32 is missing from the app bundle")
        }
        let d = try Data(contentsOf: url)
        guard d.count == 4 * count else { throw TitaNetError.contract("\(name).f32: \(d.count) bytes, expected \(4 * count)") }
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
