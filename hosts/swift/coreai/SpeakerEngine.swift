// SpeakerEngine — 16 kHz mono samples in, the 192-d TitaNet embedding out. Owns the model (Host) and the mel front
// end, off the main actor; every load and embed runs one at a time, in order. Looks for titanet.aimodel, export.json,
// mel_fb.f32 and mel_window.f32 in the app bundle (Bundle.main); add them to your target's resources.

import CoreAI
import Foundation

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

    /// Core AI specialization options for each choice
    var options: SpecializationOptions {
        switch self {
        case .cpu: return .cpuOnly
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .ane: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .auto: return .default
        }
    }
}

struct EngineInfo: Sendable, Equatable {
    var window: Int                 // F, mel frames per scan call
    var windowSeconds: Double
    var margin: Int
    var mels: Int
    var entrypoints: [String]
    var arm: String
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
}

actor SpeakerEngine {
    static let sampleRate = 16000

    private var host: Host?
    private var mel: MelFrontend?
    private var tail: Task<Void, Never>?

    func load(_ unit: ComputeUnit) async throws -> EngineInfo {
        try await serially { try await self.loadNow(unit) }
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

    private func loadNow(_ unit: ComputeUnit) async throws -> EngineInfo {
        host = nil      // release the old specialization before building the next
        let res = Bundle.main
        guard let graph = res.url(forResource: "titanet", withExtension: "aimodel"),
              let exportURL = res.url(forResource: "export", withExtension: "json") else {
            throw TitaNetError.contract("titanet.aimodel / export.json are missing from the app bundle")
        }
        let record = try JSONDecoder().decode(ExportRecord.self, from: Data(contentsOf: exportURL))
        if mel == nil {
            let fb = try Self.floats("mel_fb", count: MelFrontend.nMels * MelFrontend.bins)
            let w = try Self.floats("mel_window", count: MelFrontend.win)
            mel = MelFrontend(filterbank: fb, window: w)
        }
        let t0 = ContinuousClock.now
        let h = try await Host(graph: graph, record: record, options: unit.options)
        guard h.nMels == MelFrontend.nMels else {
            throw TitaNetError.contract("the bundle takes \(h.nMels) mels, the front end makes \(MelFrontend.nMels)")
        }
        host = h
        return EngineInfo(window: h.F, windowSeconds: Double(h.F * MelFrontend.hop) / Double(Self.sampleRate),
                          margin: h.margin, mels: h.nMels, entrypoints: record.entrypoints, arm: record.arm ?? "—",
                          architecture: AIModel.deviceArchitectureName, compute: unit, loadSeconds: seconds(since: t0))
    }

    private func embedNow(_ samples: [Float], progress: @MainActor @Sendable @escaping (String) -> Void) async throws -> Embedding {
        guard let host, let mel else { throw TitaNetError.contract("the model is not loaded") }
        await progress("Extracting log-mel features")
        let t0 = ContinuousClock.now
        let (m, frames, n) = mel(samples)
        guard n >= 1 else { throw TitaNetError.audio("The clip is too short to embed.") }
        // repack the n valid frames into a tight [80, n] buffer: the window slicer's row stride is n
        var packed = [Float](repeating: 0, count: MelFrontend.nMels * n)
        for c in 0..<MelFrontend.nMels {
            let src = c * frames, dst = c * n
            for t in 0..<n { packed[dst + t] = m[src + t] }
        }
        let featureSeconds = seconds(since: t0)
        let t1 = ContinuousClock.now
        let r = try await host.embed(mel: packed, n: n, progress: progress)
        let vector = r.emb.map { Float($0) }
        guard vector.allSatisfy(\.isFinite) else { throw TitaNetError.contract("the embedding is not finite") }
        return Embedding(vector: vector, frames: n, windows: r.windows, calls: r.calls,
                         featureSeconds: featureSeconds, modelSeconds: seconds(since: t1))
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

func seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}
