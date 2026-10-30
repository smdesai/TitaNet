// AppModel — the app's state on the main actor: the engine's status (downloading, compiling, loading, ready), the
// three clip slots (Compare A / B and the Identify probe), the enrolled speakers (Documents/speakers.json), and the
// shared recorder, player and model downloader.

import AVFoundation
import Foundation
import SwiftUI

struct Clip: Identifiable, Sendable {
    enum Source: Sendable, Equatable {
        case microphone
        case file(String)

        var label: String {
            switch self {
            case .microphone: return "Microphone"
            case .file(let name): return name
            }
        }
        var symbol: String { self == .microphone ? "mic.fill" : "doc.fill" }
    }

    let id = UUID()
    let samples: [Float]
    let envelope: [Float]
    let source: Source
    let embedding: Embedding
    let trimmed: Bool

    var seconds: Double { Double(samples.count) / AudioLoader.rate }
}

@MainActor @Observable
final class ClipSlot: Identifiable {
    enum State {
        case empty
        case recording
        case processing(String)
        case ready(Clip)
        case failed(String)
    }

    let id = UUID()
    var state: State = .empty
    /// the envelope of the clip being processed, so the card can keep showing it
    var preview: [Float] = []

    var clip: Clip? { if case .ready(let c) = state { c } else { nil } }
    var isBusy: Bool {
        switch state {
        case .recording, .processing: true
        default: false
        }
    }
}

struct Speaker: Identifiable, Codable, Equatable {
    var id = UUID()
    var name: String
    var hue: Double
    var embedding: [Float]
    var seconds: Double
    var created = Date()

    var color: Color { Color(hue: hue, saturation: 0.62, brightness: 0.98) }
    var initials: String {
        let parts = name.split(separator: " ").prefix(2)
        let s = parts.compactMap(\.first).map(String.init).joined()
        return s.isEmpty ? "?" : s.uppercased()
    }
}

@MainActor @Observable
final class AppModel {
    enum EngineStatus: Equatable {
        case downloading(Runtime)       // from Hugging Face; progress is on `downloader`
        case compiling(Runtime)         // a Core ML variant's packages, once, for this device
        case loading                    // into the runtime (Core AI specializes the bundle on its first load)
        case ready(EngineInfo)
        case failed(String)
    }

    static let maxSeconds = 180.0
    static let minSeconds = 0.5

    private(set) var status: EngineStatus = .loading
    var runtime: Runtime {
        didSet {
            guard runtime != oldValue else { return }
            UserDefaults.standard.set(runtime.rawValue, forKey: "runtime")
            Task { await load() }
        }
    }
    var compute: ComputeUnit {
        didSet {
            guard compute != oldValue else { return }
            UserDefaults.standard.set(compute.rawValue, forKey: "compute")
            Task { await load() }
        }
    }
    var threshold: Double {
        didSet { UserDefaults.standard.set(threshold, forKey: "threshold") }
    }

    let a = ClipSlot()
    let b = ClipSlot()
    let probe = ClipSlot()
    private(set) var speakers: [Speaker] = []
    var alert: String?

    let recorder = Recorder()
    let player = Player()
    @ObservationIgnored let downloader = ModelDownloader()
    private let engine = SpeakerEngine()
    /// Loads run one after another (the downloader runs one fetch at a time); the last asked for wins.
    @ObservationIgnored private var loadTask: Task<Void, Never>?

    init() {
        runtime = Runtime(rawValue: UserDefaults.standard.string(forKey: "runtime") ?? "") ?? .coreAI
        compute = ComputeUnit(rawValue: UserDefaults.standard.string(forKey: "compute") ?? "") ?? .gpu
        let t = UserDefaults.standard.double(forKey: "threshold")
        threshold = t > 0 ? t : Scoring.threshold
        speakers = Self.readSpeakers()
        recorder.onLimit = { [weak self] in
            guard let self, let slot = self.recorder.slot else { return }
            Task { await self.stopRecording(slot) }
        }
    }

    var info: EngineInfo? { if case .ready(let i) = status { i } else { nil } }

    /// Downloading or compiling: the runtime pickers wait for it.
    var isFetching: Bool {
        switch status {
        case .downloading, .compiling: true
        default: false
        }
    }

    /// Downloads the selected runtime's model if this device does not have it yet, then loads it.
    func load() async {
        let previous = loadTask
        let task = Task { await previous?.value; await self.loadNow() }
        loadTask = task
        await task.value
    }

    private func loadNow() async {
        let rt = runtime, cu = compute
        do {
            if !ModelStore.isReady(rt) { status = .downloading(rt) }
            let folder = try await ModelStore.ensure(rt, with: downloader) { self.status = .compiling(rt) }
            status = .loading
            status = .ready(try await engine.load(rt, cu, from: folder))
        } catch {
            status = .failed("\(error)")
        }
        print("[studio] \(rt.label) · \(cu.label): \(status)")
    }

    /// `-selftest YES` at launch: embed two synthetic signals (4 s, one window; 65 s, sixteen) with every runtime and
    /// log each one's cosine to Core AI on the GPU, and its time. Checks the wiring of both runtimes on a device
    /// without touching the screen; accuracy against NeMo is the gates' job (LeanGate, MLGate).
    func selfTest() async {
        let configs: [(Runtime, ComputeUnit)] = [(.coreAI, .gpu), (.coreMLFP32, .gpu), (.coreMLFP32, .cpu),
                                                 (.coreMLFP16, .ane), (.coreMLFP16, .gpu), (.coreAI, .ane)]
        let clips = [("4s", Self.syntheticVoice(seconds: 4, seed: 1)), ("65s", Self.syntheticVoice(seconds: 65, seed: 2))]
        var reference: [String: [Float]] = [:]
        for (rt, cu) in configs {
            do {
                if !ModelStore.isReady(rt) { status = .downloading(rt) }
                let folder = try await ModelStore.ensure(rt, with: downloader) { self.status = .compiling(rt) }
                let info = try await engine.load(rt, cu, from: folder)
                for (name, x) in clips {
                    let e = try await engine.embed(x) { _ in }
                    if reference[name] == nil { reference[name] = e.vector }
                    let cos = Scoring.cosine(e.vector, reference[name]!)
                    print(String(format: "[studio] selftest %@ · %@ %@: cos_vs_coreai_gpu %.9f model %.3f s, %d calls, load %.2f s",
                                 rt.label, cu.short, name, cos, e.modelSeconds, e.calls, info.loadSeconds))
                }
            } catch {
                print("[studio] selftest \(rt.label) · \(cu.short): FAILED \(error)")
            }
        }
        print("[studio] selftest done")
        await load()
    }

    /// A deterministic voiced signal: a glottal-like harmonic series on a gliding pitch, with formant-ish weighting
    /// and a little noise, gated into syllables. Not speech, but it exercises every part of the model.
    static func syntheticVoice(seconds: Double, seed: UInt64) -> [Float] {
        let n = Int(seconds * AudioLoader.rate)
        var x = [Float](repeating: 0, count: n)
        var s = seed &* 6364136223846793005 &+ 1442695040888963407
        var phase = 0.0
        for i in 0..<n {
            let t = Double(i) / AudioLoader.rate
            let f0 = 120 + 30 * sin(2 * .pi * 0.7 * t) + 15 * sin(2 * .pi * 2.3 * t)
            phase += 2 * .pi * f0 / AudioLoader.rate
            var v = 0.0
            for h in 1...24 {
                let fh = f0 * Double(h)
                let formant = exp(-pow((fh - 700) / 300, 2)) + 0.6 * exp(-pow((fh - 1200) / 400, 2)) + 0.3 * exp(-pow((fh - 2600) / 500, 2))
                v += (0.15 + formant) / Double(h) * sin(Double(h) * phase)
            }
            let gate = max(0, sin(2 * .pi * 3.1 * t))
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let noise = Double(Int64(bitPattern: s) >> 11) / Double(1 << 52)
            x[i] = Float(0.08 * gate * v + 0.003 * noise)
        }
        return x
    }

    // MARK: clips

    func toggleRecording(_ slot: ClipSlot) async {
        if recorder.slot === slot {
            await stopRecording(slot)
            return
        }
        if let other = recorder.slot { await stopRecording(other) }
        player.stop()
        do {
            try await recorder.start(slot)
            slot.state = .recording
        } catch {
            alert = "\(error)"
        }
    }

    func stopRecording(_ slot: ClipSlot) async {
        guard recorder.slot === slot else { return }
        let samples = recorder.stop()
        await ingest(samples, source: .microphone, into: slot)
    }

    func importFile(_ url: URL, into slot: ClipSlot) async {
        player.stop()
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        slot.preview = []
        slot.state = .processing("Decoding \(url.lastPathComponent)")
        do {
            let samples = try await Task.detached { try AudioLoader.load(url) }.value
            await ingest(samples, source: .file(url.lastPathComponent), into: slot)
        } catch {
            slot.state = .failed("\(error)")
        }
    }

    func ingest(_ raw: [Float], source: Clip.Source, into slot: ClipSlot) async {
        let cap = Int(Self.maxSeconds * AudioLoader.rate)
        let samples = raw.count > cap ? Array(raw[0..<cap]) : raw
        guard Double(samples.count) / AudioLoader.rate >= Self.minSeconds else {
            slot.state = .failed("That clip is under half a second. Speak for at least a couple of seconds.")
            return
        }
        let envelope = Self.envelope(samples)
        slot.preview = envelope
        slot.state = .processing("Waiting for the model")
        do {
            // a clip recorded before any model is in (first launch: still downloading) waits for it
            if info == nil { await loadTask?.value }
            guard info != nil else { throw StudioError.contract("The model is not available. See the Model tab.") }
            let e = try await engine.embed(samples) { [weak slot] phase in
                if case .processing = slot?.state { slot?.state = .processing(phase) }
            }
            slot.state = .ready(Clip(samples: samples, envelope: envelope, source: source, embedding: e,
                                     trimmed: raw.count > cap))
        } catch {
            slot.state = .failed("\(error)")
        }
    }

    func clear(_ slot: ClipSlot) {
        if recorder.slot === slot { _ = recorder.stop() }
        if let c = slot.clip, player.playing == c.id { player.stop() }
        slot.state = .empty
        slot.preview = []
    }

    // MARK: scores

    func score(_ x: ClipSlot, _ y: ClipSlot) -> Double? {
        guard let p = x.clip, let q = y.clip else { return nil }
        return Scoring.score(p.embedding.vector, q.embedding.vector)
    }

    /// Enrolled speakers against the probe, best first.
    var ranking: [(speaker: Speaker, score: Double)] {
        guard let p = probe.clip else { return [] }
        return speakers.map { ($0, Scoring.score(p.embedding.vector, $0.embedding)) }.sorted { $0.score > $1.score }
    }

    // MARK: speakers

    func enroll(name: String, clip: Clip) {
        let used = Set(speakers.map { Int($0.hue * 12) })
        let slot = (0..<12).first { !used.contains($0) } ?? speakers.count % 12
        speakers.append(Speaker(name: name, hue: Double(slot) / 12, embedding: clip.embedding.vector, seconds: clip.seconds))
        writeSpeakers()
    }

    func delete(_ speaker: Speaker) {
        speakers.removeAll { $0.id == speaker.id }
        writeSpeakers()
    }

    private static var speakersURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("speakers.json")
    }

    private static func readSpeakers() -> [Speaker] {
        guard let d = try? Data(contentsOf: speakersURL) else { return [] }
        return (try? JSONDecoder().decode([Speaker].self, from: d)) ?? []
    }

    private func writeSpeakers() {
        do { try JSONEncoder().encode(speakers).write(to: Self.speakersURL, options: .atomic) } catch {
            alert = "Could not save enrolled voices: \(error.localizedDescription)"
        }
    }

    /// Peak-normalized RMS in `bins` slices, for the waveform drawings.
    static func envelope(_ x: [Float], bins: Int = 72) -> [Float] {
        guard !x.isEmpty else { return [] }
        let per = max(1, x.count / bins)
        var out: [Float] = []
        var i = 0
        while i < x.count && out.count < bins {
            let end = min(x.count, i + per)
            var ss: Float = 0
            for j in i..<end { ss += x[j] * x[j] }
            out.append((ss / Float(end - i)).squareRoot())
            i = end
        }
        let peak = out.max() ?? 0
        return peak > 0 ? out.map { $0 / peak } : out
    }
}

// MARK: - recorder and player

@MainActor @Observable
final class Recorder {
    private(set) var slot: ClipSlot?
    private(set) var levels: [Float] = []
    private(set) var elapsed: Double = 0
    @ObservationIgnored var onLimit: (() -> Void)?
    @ObservationIgnored private var capture: MicCapture?
    @ObservationIgnored private var ticker: Task<Void, Never>?

    static let limit = 60.0
    static let bars = 48

    func start(_ slot: ClipSlot) async throws {
        guard await AVAudioApplication.requestRecordPermission() else {
            throw StudioError.audio("Microphone access is off. Turn it on in Settings › TitaNet Studio.")
        }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)
        let c = MicCapture()
        try c.start()
        capture = c
        self.slot = slot
        levels = []
        elapsed = 0
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(40))
                guard let self, let c = self.capture else { return }
                let (lv, t) = c.snapshot(lastLevels: Self.bars)
                self.levels = lv
                self.elapsed = t
                if t >= Self.limit { self.onLimit?(); return }
            }
        }
    }

    func stop() -> [Float] {
        ticker?.cancel()
        ticker = nil
        let samples = capture?.stop() ?? []
        capture = nil
        slot = nil
        return samples
    }
}

@MainActor @Observable
final class Player {
    private(set) var playing: UUID?
    private(set) var progress: Double = 0
    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var ticker: Task<Void, Never>?

    func toggle(_ clip: Clip) {
        if playing == clip.id { stop(); return }
        stop()
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
            let p = try AVAudioPlayer(data: WAV.data(clip.samples))
            p.play()
            player = p
            playing = clip.id
            ticker = Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(30))
                    guard let self, let p = self.player else { return }
                    if !p.isPlaying { self.stop(); return }
                    self.progress = p.duration > 0 ? p.currentTime / p.duration : 0
                }
            }
        } catch {
            stop()
        }
    }

    func stop() {
        ticker?.cancel()
        ticker = nil
        player?.stop()
        player = nil
        playing = nil
        progress = 0
    }
}
