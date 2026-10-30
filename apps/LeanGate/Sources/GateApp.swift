// LeanGate — the device side of the titanet-lean memory gate, headless, with no input from the screen.
//
// It answers one question: what is the authoritative iOS `phys_footprint` of an exported titanet-lean bundle while
// it runs the host algorithm of tnl/runtime.py over utterances of a given length? The arithmetic of the host
// reductions is NOT what is measured here (the mel is synthetic): the call COUNT, tensor shapes and dtypes are, so
// that the memory and latency profile is the one a shipping host would see.
//
// Library/Application Support/LeanAssets/ (pushed with device.sh push):
//   bundles/<name>/{titanet.aimodel, export.json}  as export/export_coreai.py writes them, and
//   gate.json: {"tag": "v2_scan_f512", "bundle": "bundles/v2_scan_f512", "compute": "gpu|ane|cpu|default",
//               "frames": [408, 828, 2052, 6472], "iters": 3}
// Documents/titanet_gate/ (pulled with device.sh run):
//   status.json   phase, architecture, thermal state, launch/loaded/per-n footprint and the PEAK footprint_mb;
//                 rewritten as the run goes so a driver can poll it. phase reaches "finished" or "failed".
//
// Memory is read exactly as TitaNetGate reads it: TASK_VM_INFO -> phys_footprint / 1048576.

import CoreAI
import Darwin
import Foundation
import SwiftUI
import UIKit

@main
struct GateApp: App {
    @State private var gate = GateRun()

    var body: some Scene {
        WindowGroup {
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    Text("LeanGate").font(.headline)
                    Text(gate.line).font(.system(.footnote, design: .monospaced))
                }
                .padding()
            }
            .task {
                // keep the screen on: a locked phone suspends the app, and a suspended app may not use the GPU
                UIApplication.shared.isIdleTimerDisabled = true
                await gate.run()
            }
        }
    }
}

// MARK: - configuration and the export record

struct GateConfig: Decodable {
    var tag: String?
    var bundle: String?            // "bundles/<name>": a directory with titanet.aimodel + export.json
    var compute: String?
    var frames: [Int]?             // synthetic mode: utterance lengths in mel frames to exercise
    var iters: Int?
    var manifest: String?          // accuracy mode: "audio/manifest.jsonl". Present => accuracy INSTEAD of frames
    var limit: Int?                //   how many of its rows to run (default all)
}

/// One line of audio/manifest.jsonl. `mel` is [80, frames] row-major float32; the utterance is its first `mel_len`
/// frames. `emb` is NeMo's own [192] float32 embedding, the reference this mode is measured against.
struct ManifestRow: Decodable {
    var id: String
    var mel: String
    var mel_len: Int
    var frames: Int
    var emb: String
    var seconds: Double
}

/// The fields of export.json this app needs (export/export_coreai.py writes the rest).
struct ExportRecord: Decodable {
    struct Signature: Decodable { var inputs: [String]; var outputs: [String] }
    var shape: Int
    var margin: Int?
    var entrypoints: [String]
    var compute_dtype: String
    var signature: [String: Signature]?
}

enum GateError: Error, CustomStringConvertible {
    case contract(String)
    var description: String { switch self { case .contract(let s): return s } }
}

// MARK: - the run

@MainActor @Observable
final class GateRun {
    var line = "starting"
    @ObservationIgnored private var status: [(String, Any)] = []
    @ObservationIgnored private var peakMB = 0.0
    @ObservationIgnored private let fm = FileManager.default
    @ObservationIgnored private let assets = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("LeanAssets")
    @ObservationIgnored private let out = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("titanet_gate")

    func run() async {
        do {
            try fm.createDirectory(at: out, withIntermediateDirectories: true)
            let cfg = try JSONDecoder().decode(GateConfig.self, from: Data(contentsOf: assets.appendingPathComponent("gate.json")))
            let compute = cfg.compute ?? "default"
            let dir = assets.appendingPathComponent(cfg.bundle ?? "bundles/titanet")
            let tag = cfg.tag ?? dir.lastPathComponent
            let frames = cfg.frames ?? [408, 828, 2052, 6472]
            let iters = max(1, cfg.iters ?? 3)
            let rec = try JSONDecoder().decode(ExportRecord.self, from: Data(contentsOf: dir.appendingPathComponent("export.json")))
            let graph = try Bundles.resolve("titanet", in: dir)

            note("phase", "load", ["tag": tag, "bundle": cfg.bundle ?? dir.lastPathComponent, "compute": compute,
                                   "arch": AIModel.deviceArchitectureName, "graph": graph.lastPathComponent,
                                   "shape": rec.shape, "margin": rec.margin ?? 46,
                                   "entrypoints": rec.entrypoints, "compute_dtype": rec.compute_dtype,
                                   "frames": frames, "iters": iters])

            // 2. the footprint BEFORE anything is loaded
            let launchMB = Self.footprintMB()
            mark(launchMB)

            // 3. load the bundle with the Core AI Swift runtime
            let t0 = ContinuousClock.now
            let host = try await Host(bundle: dir, graph: graph, record: rec, compute: compute)
            let loadS = secondsSince(t0)
            let loadedMB = Self.footprintMB()
            mark(loadedMB)
            note("phase", "loaded", ["launch_mb": launchMB, "loaded_mb": loadedMB, "load_s": loadS])

            // 4. accuracy mode when gate.json names a manifest, else the synthetic frames sweep
            if let manifest = cfg.manifest {
                try await accuracy(host, manifest: assets.appendingPathComponent(manifest), limit: cfg.limit)
                note("phase", "finished")
                return
            }
            for n in frames {
                note("phase", "n\(n)")
                let mel = Self.syntheticMel(frames: n, mels: host.nMels, seed: UInt64(n))
                var secs: [Double] = []
                var calls = 0
                var wins = 0
                for i in 0..<iters {
                    let t = ContinuousClock.now
                    let r = try await host.embed(mel: mel, n: n) { phase in
                        self.note("phase", "n\(n) iter \(i + 1)/\(iters) \(phase)")
                    }
                    secs.append(secondsSince(t))
                    calls = r.calls
                    wins = r.windows
                    mark(Self.footprintMB())
                }
                secs.sort()
                note("n\(n)_s", secs[secs.count / 2], ["n\(n)_mb": Self.footprintMB(), "n\(n)_windows": wins,
                                                       "n\(n)_calls": calls, "n\(n)_s_all": secs])
            }
            note("phase", "finished")
        } catch {
            note("phase", "failed", ["error": "\(error)"])
        }
    }

    // MARK: - accuracy mode

    /// Every manifest row, in order: the real mel through the same host algorithm, against NeMo's own embedding.
    private func accuracy(_ host: Host, manifest: URL, limit: Int?) async throws {
        let dir = manifest.deletingLastPathComponent()
        let text = try String(contentsOf: manifest, encoding: .utf8)
        let dec = JSONDecoder()
        var rows: [ManifestRow] = []
        for line in text.split(separator: "\n") where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            rows.append(try dec.decode(ManifestRow.self, from: Data(line.utf8)))
        }
        if let limit, limit > 0, limit < rows.count { rows = Array(rows[0..<limit]) }
        guard !rows.isEmpty else { throw GateError.contract("\(manifest.lastPathComponent): no rows") }

        var lines: [String] = []
        var cosines: [Double] = []
        var latencies: [Double] = []
        var callsMax = 0
        var allFinite = true

        for (i, row) in rows.enumerated() {
            note("phase", "acc \(i + 1)/\(rows.count)")
            // mel [80, row.frames] row-major: element (c, t) is at c * row.frames + t. Repack the utterance's first
            // mel_len frames into a tight [80, n] buffer so the window slicer's row stride is n, as it is for
            // synthetic mel. Getting this transposed would show up as cos ~ 0.
            let n = row.mel_len
            guard n >= 1 else { throw GateError.contract("\(row.id): mel_len \(n)") }
            let raw = try Self.float32s(at: dir.appendingPathComponent(row.mel))
            guard raw.count == host.nMels * row.frames else {
                throw GateError.contract("\(row.id): mel has \(raw.count) floats, expected \(host.nMels) x \(row.frames)")
            }
            guard n <= row.frames else { throw GateError.contract("\(row.id): mel_len \(n) > frames \(row.frames)") }
            var mel = [Float](repeating: 0, count: host.nMels * n)
            for c in 0..<host.nMels {
                let src = c * row.frames, dst = c * n
                for t in 0..<n { mel[dst + t] = raw[src + t] }
            }
            let reference = try Self.float32s(at: dir.appendingPathComponent(row.emb))

            let t = ContinuousClock.now
            let r = try await host.embed(mel: mel, n: n) { phase in
                self.note("phase", "acc \(i + 1)/\(rows.count) \(phase)")
            }
            let latency = secondsSince(t)
            mark(Self.footprintMB())

            guard reference.count == r.emb.count else {
                throw GateError.contract("\(row.id): reference has \(reference.count) values, the graph returned \(r.emb.count)")
            }
            let finite = r.emb.allSatisfy { $0.isFinite }
            let cos = Self.cosine(r.emb, reference.map { Double($0) })
            if !finite || !cos.isFinite { allFinite = false }
            let path = (n <= host.F && host.record.entrypoints.contains("once")) ? "once" : "scan"
            cosines.append(cos)
            latencies.append(latency)
            callsMax = max(callsMax, r.calls)
            lines.append(jsonLine(["id": row.id, "seconds": row.seconds, "n": n, "path": path, "calls": r.calls,
                                   "cos": cos, "latency_s": latency, "finite": finite]))
            // rewritten every utterance, so a run that stalls still leaves everything it finished
            try (lines.joined(separator: "\n") + "\n").write(to: out.appendingPathComponent("accuracy.jsonl"),
                                                              atomically: true, encoding: .utf8)

            let sorted = cosines.sorted()
            note("phase", "acc \(i + 1)/\(rows.count)",
                 ["n": cosines.count, "cos_min": sorted[0], "cos_mean": cosines.reduce(0, +) / Double(cosines.count),
                  "cos_p10": sorted[sorted.count / 10], "all_finite": allFinite, "calls_max": callsMax,
                  "latency_mean_s": latencies.reduce(0, +) / Double(latencies.count),
                  "latency_max_s": latencies.max() ?? 0])
        }
    }

    /// A little-endian float32 file as [Float]. ARM is little-endian, so this is the file's own byte order.
    static func float32s(at url: URL) throws -> [Float] {
        let d = try Data(contentsOf: url)
        guard d.count % 4 == 0 else { throw GateError.contract("\(url.lastPathComponent): \(d.count) bytes is not a float32 count") }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<min(a.count, b.count) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let d = (na * nb).squareRoot()
        return d == 0 ? 0 : dot / d
    }

    /// A fixed pseudo-random mel [mels][frames] (feature-major, row = one mel bin). Accuracy is not measured.
    static func syntheticMel(frames n: Int, mels: Int, seed: UInt64) -> [Float] {
        var s = seed &* 6364136223846793005 &+ 1442695040888963407
        var x = [Float](repeating: 0, count: mels * n)
        for i in 0..<x.count {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            x[i] = Float(Double((s >> 33) & 0xFFFFFF) / 16777216.0) * 2 - 1
        }
        return x
    }

    private func mark(_ mb: Double) { peakMB = max(peakMB, mb) }

    private func note(_ key: String, _ value: Any, _ extra: [String: Any] = [:]) {
        mark(Self.footprintMB())
        status.removeAll { $0.0 == key || extra[$0.0] != nil || ["thermal", "footprint_mb", "current_mb", "available_mb"].contains($0.0) }
        status.append((key, value))
        for (k, v) in extra.sorted(by: { $0.key < $1.key }) { status.append((k, v)) }
        let current = Self.footprintMB()
        mark(current)
        status += [("thermal", Self.thermal()), ("footprint_mb", peakMB), ("current_mb", current),
                   ("available_mb", Double(os_proc_available_memory()) / 1_048_576)]
        line = status.map { "\($0.0): \($0.1)" }.joined(separator: "\n")
        var obj: [String: Any] = [:]
        for (k, v) in status { obj[k] = v }
        try? jsonLine(obj).write(to: out.appendingPathComponent("status.json"), atomically: true, encoding: .utf8)
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }
}

// MARK: - the host algorithm (a transcription of tnl/runtime.py)

/// [(start, keep_lo, keep_hi)] in global frame coordinates: interiors tile [0, n) exactly once. tnl/stream.windows.
func windows(_ n: Int, _ F: Int, _ margin: Int) throws -> [(s: Int, lo: Int, hi: Int)] {
    if n <= F { return [(0, 0, n)] }
    let step = F - 2 * margin
    guard step > 0 else { throw GateError.contract("window \(F) is too small for a \(margin)-frame margin on each side") }
    var out: [(Int, Int, Int)] = []
    var s = 0
    while true {
        let end = s + F
        out.append((s, s == 0 ? 0 : s + margin, end >= n ? n : end - margin))
        if end >= n { return out }
        s += step
    }
}

/// One loaded bundle, and the seven-pass scan of tnl/runtime.py. Host accumulation is Double, as there.
final class Host {
    static let stdEps = 1e-10
    let record: ExportRecord
    let F: Int
    let margin: Int
    let nMels: Int
    let cOut: Int
    private let model: AIModel
    private let scan: InferenceFunction?
    private let head: InferenceFunction?
    private let once: InferenceFunction?
    private let scanIn: [String: NDArrayDescriptor]
    private let headIn: [String: NDArrayDescriptor]
    private let onceIn: [String: NDArrayDescriptor]
    private let seSizes: [Int]
    private let scanOutputs: [String]

    init(bundle: URL, graph: URL, record: ExportRecord, compute: String) async throws {
        // Everything is computed into locals first: a nested function or closure that touched `self` here would
        // capture it before all of its members were initialized, which Swift 6 refuses.
        let F = record.shape
        let m = try await AIModel(contentsOf: graph, options: Compute.options(compute))
        let names = Set(m.functionNames)
        let hasScan = record.entrypoints.contains("scan") && names.contains("scan")
        let hasOnce = record.entrypoints.contains("once") && names.contains("once")
        guard hasScan || hasOnce else {
            throw GateError.contract("\(graph.lastPathComponent): neither scan nor once; functions \(names.sorted())")
        }
        let scanD = hasScan ? try Host.descriptors(m, "scan") : [:]
        let onceD = hasOnce ? try Host.descriptors(m, "once") : [:]
        let headD = hasScan ? try Host.descriptors(m, "head") : [:]

        guard let melD = scanD["mel"] ?? onceD["mel"], melD.shape.count == 3, melD.shape[2] == F else {
            throw GateError.contract("mel input is \((scanD["mel"] ?? onceD["mel"])?.shape.description ?? "missing"), "
                + "expected [1, mels, \(F)]")
        }
        var c = 0
        var se: [Int] = []
        if hasScan {
            guard let mean = scanD["mean"], mean.shape.count == 2 else { throw GateError.contract("scan.mean is missing") }
            c = mean.shape[1]
            for j in 0..<5 {
                guard let d = scanD["s\(j)"], d.shape.count == 2 else { throw GateError.contract("scan.s\(j) is missing") }
                se.append(d.shape[1])
            }
        }

        self.record = record
        self.F = F
        self.margin = record.margin ?? 46
        self.model = m
        // the names `scan` returns, as export.json's signature records them
        self.scanOutputs = record.signature?["scan"]?.outputs
            ?? ["se0", "se1", "se2", "se3", "se4", "x_sum", "x_sqsum", "lmax", "den", "ax", "axx"]
        self.scanIn = scanD
        self.onceIn = onceD
        self.headIn = headD
        self.scan = hasScan ? try Host.load(m, "scan") : nil
        self.head = hasScan ? try Host.load(m, "head") : nil
        self.once = hasOnce ? try Host.load(m, "once") : nil
        self.nMels = melD.shape[1]
        self.cOut = c
        self.seSizes = se
    }

    /// Every NDArray input of `name`, by name, as the graph itself declares it (shape and scalar type).
    private static func descriptors(_ model: AIModel, _ name: String) throws -> [String: NDArrayDescriptor] {
        guard let md = model.functionDescriptor(for: name) else { throw GateError.contract("\(name): no function descriptor") }
        var out: [String: NDArrayDescriptor] = [:]
        for i in md.inputNames {
            guard let d = ND.descriptor(md.inputDescriptor(of: i)) else {
                throw GateError.contract("\(name).\(i) is not an NDArray input")
            }
            out[i] = d
        }
        return out
    }

    private static func load(_ model: AIModel, _ name: String) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: name) else { throw GateError.contract("\(name) did not load") }
        return fn
    }

    /// mel: [nMels * n] feature-major with the utterance's n valid frames first. emb [192] is returned unchecked.
    func embed(mel: [Float], n: Int, progress: @MainActor @escaping (String) -> Void) async throws
        -> (emb: [Double], calls: Int, windows: Int) {
        guard n >= 1 else { throw GateError.contract("no valid mel frame") }
        if n <= F, let once {
            let (mw, vw, _) = window(mel, n: n, s: 0, keep: (0, n))
            var inputs: [String: NDArray] = [:]
            inputs["mel"] = nd(mw, onceIn["mel"]!)
            inputs["valid"] = nd(vw, onceIn["valid"]!)
            var o = try await once.run(inputs: inputs)
            guard let e = o.remove("emb")?.ndArray else { throw GateError.contract("once: no emb output") }
            return (ND.doubles(e), 1, 1)
        }
        guard let scan, let head else { throw GateError.contract("\(n) frames need the scan path, which this bundle lacks") }
        let wins = try windows(n, F, margin)
        let nInv = 1.0 / Double(n)
        var calls = 0
        var sums = seSizes.map { [Double](repeating: 0, count: $0) }
        let zeroMean = [Double](repeating: 0, count: cOut)
        let unitStd = [Double](repeating: 1, count: cOut)

        // 1. the SqueezeExcite gate fixed point: five sweeps, the sums fed back as s0..s4
        for sweep in 0..<5 {
            await progress("gate sweep \(sweep + 1)/5")
            var new = seSizes.map { [Double](repeating: 0, count: $0) }
            for w in wins {
                let o = try await call(scan, mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: zeroMean, std: unitStd)
                calls += 1
                for j in 0..<5 {
                    guard let se = o["se\(j)"] else { throw GateError.contract("scan: no se\(j) output") }
                    for k in 0..<min(new[j].count, se.count) { new[j][k] += se[k] }
                }
            }
            sums = new
        }

        // 2. the pooling mean / std of the encoder output
        await progress("pool pass")
        var sx = [Double](repeating: 0, count: cOut)
        var sxx = [Double](repeating: 0, count: cOut)
        for w in wins {
            let o = try await call(scan, mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: zeroMean, std: unitStd)
            calls += 1
            guard let a = o["x_sum"], let b = o["x_sqsum"] else { throw GateError.contract("scan: no x_sum / x_sqsum") }
            for k in 0..<cOut { sx[k] += a[k]; sxx[k] += b[k] }
        }
        var pmean = [Double](repeating: 0, count: cOut)
        var pstd = [Double](repeating: 0, count: cOut)
        for k in 0..<cOut {
            pmean[k] = sx[k] / Double(n)
            pstd[k] = (max(sxx[k] / Double(n) - pmean[k] * pmean[k], Host.stdEps)).squareRoot()
        }

        // 3. the attention softmax over time, combined across windows by running maximum
        await progress("attention pass")
        var mx: [Double]? = nil
        var den = [Double](repeating: 0, count: cOut)
        var ax = [Double](repeating: 0, count: cOut)
        var axx = [Double](repeating: 0, count: cOut)
        for w in wins {
            let o = try await call(scan, mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: pmean, std: pstd)
            calls += 1
            guard let lmax = o["lmax"], let d = o["den"], let a = o["ax"], let aa = o["axx"] else {
                throw GateError.contract("scan: no lmax / den / ax / axx")
            }
            var scale = [Double](repeating: 0, count: cOut)
            var f = [Double](repeating: 1, count: cOut)
            if let m = mx {
                var mNew = [Double](repeating: 0, count: cOut)
                for k in 0..<cOut {
                    mNew[k] = max(m[k], lmax[k])
                    scale[k] = exp(m[k] - mNew[k])
                    f[k] = exp(lmax[k] - mNew[k])
                }
                mx = mNew
            } else {
                mx = lmax
            }
            for k in 0..<cOut {
                den[k] = den[k] * scale[k] + d[k] * f[k]
                ax[k] = ax[k] * scale[k] + a[k] * f[k]
                axx[k] = axx[k] * scale[k] + aa[k] * f[k]
            }
        }
        var mu = [Double](repeating: 0, count: cOut)
        var sg = [Double](repeating: 0, count: cOut)
        for k in 0..<cOut {
            let q = den[k] == 0 ? 0 : ax[k] / den[k]
            mu[k] = q
            let v = den[k] == 0 ? Host.stdEps : axx[k] / den[k] - q * q
            sg[k] = max(v, Host.stdEps).squareRoot()
        }

        await progress("head")
        var inputs: [String: NDArray] = [:]
        inputs["mu"] = nd(mu.map { Float($0) }, headIn["mu"]!)
        inputs["sg"] = nd(sg.map { Float($0) }, headIn["sg"]!)
        var o = try await head.run(inputs: inputs)
        calls += 1
        guard let e = o.remove("emb")?.ndArray else { throw GateError.contract("head: no emb output") }
        return (ND.doubles(e), calls, wins.count)
    }

    /// One `scan` call with the eleven inputs tnl/runtime._scan feeds.
    private func call(_ scan: InferenceFunction, mel: [Float], n: Int, w: (s: Int, lo: Int, hi: Int), nInv: Double,
                      sums: [[Double]], mean: [Double], std: [Double]) async throws -> [String: [Double]] {
        let (mw, vw, kw) = window(mel, n: n, s: w.s, keep: (w.lo, w.hi))
        var inputs: [String: NDArray] = [:]
        inputs["mel"] = nd(mw, scanIn["mel"]!)
        inputs["valid"] = nd(vw, scanIn["valid"]!)
        inputs["keep"] = nd(kw, scanIn["keep"]!)
        inputs["n_inv"] = nd([Float(nInv)], scanIn["n_inv"]!)
        inputs["mean"] = nd(mean.map { Float($0) }, scanIn["mean"]!)
        inputs["std"] = nd(std.map { Float($0) }, scanIn["std"]!)
        for j in 0..<5 { inputs["s\(j)"] = nd(sums[j].map { Float($0) }, scanIn["s\(j)"]!) }
        var out = try await scan.run(inputs: inputs)
        var r: [String: [Double]] = [:]
        for name in scanOutputs {
            guard let a = out.remove(name)?.ndArray else { continue }
            r[name] = ND.doubles(a)
        }
        return r
    }

    /// mel [nMels, n] -> (mel_w [1, nMels, F], valid_w [1, 1, F], keep_w [1, 1, F]). tnl/runtime._window.
    private func window(_ mel: [Float], n: Int, s: Int, keep: (Int, Int)) -> ([Float], [Float], [Float]) {
        var mw = [Float](repeating: 0, count: nMels * F)
        let take = max(0, min(F, n - s))
        if take > 0 {
            for c in 0..<nMels {
                let src = c * n + s
                for f in 0..<take { mw[c * F + f] = mel[src + f] }
            }
        }
        var vw = [Float](repeating: 0, count: F)
        for f in 0..<take { vw[f] = 1 }
        var kw = [Float](repeating: 0, count: F)
        let lo = max(0, keep.0 - s), hi = min(F, keep.1 - s)
        if hi > lo { for f in lo..<hi { kw[f] = 1 } }
        return (mw, vw, kw)
    }

    /// Input dtype comes from the graph's own descriptor, which export.json `compute_dtype` set.
    private func nd(_ v: [Float], _ d: NDArrayDescriptor) -> NDArray {
        d.scalarType == .float16 ? ND.make(v.map { Float16($0) }, d) : ND.make(v, d)
    }
}

// MARK: - plumbing lifted from TitaNetKit (ND.swift, Bundles.swift, Batch.jsonLine)

enum ND {
    static func descriptor(_ value: InferenceValue.Descriptor?) -> NDArrayDescriptor? {
        guard case .ndArray(let d) = value else { return nil }
        return d
    }

    /// A new array for `descriptor`, filled row-major from `values`.
    static func make<T: BitwiseCopyable>(_ values: [T], _ descriptor: NDArrayDescriptor) -> NDArray {
        var array = NDArray(descriptor: descriptor)
        var view = array.mutableView(as: T.self)
        view.copyElements(fromContentsOf: values)
        return array
    }

    /// Row-major copy of an array's scalars, honoring its strides (in elements).
    static func read<T: BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> [T] {
        let shape = array.shape
        let count = shape.reduce(1, *)
        return array.view(as: T.self).withUnsafePointer { ptr, _, strides in
            var expected = 1
            var contiguous = true
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                if shape[d] > 1 && strides[d] != expected {
                    contiguous = false
                    break
                }
                expected *= shape[d]
            }
            if contiguous { return Array(UnsafeBufferPointer(start: ptr, count: count)) }
            var out = [T]()
            out.reserveCapacity(count)
            var index = [Int](repeating: 0, count: shape.count)
            for _ in 0..<count {
                var offset = 0
                for d in 0..<shape.count { offset += index[d] * strides[d] }
                out.append(ptr[offset])
                var d = shape.count - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
            return out
        }
    }

    static func doubles(_ a: NDArray) -> [Double] {
        a.scalarType == .float16 ? read(a, as: Float16.self).map { Double($0) } : read(a, as: Float.self).map { Double($0) }
    }
}

enum Bundles {
    /// `<name>.<arch>.aimodelc` / `<name>.aimodelc` compiled for this device, else a JIT `<name>.aimodel`.
    static func resolve(_ name: String, in directory: URL,
                        deviceArchitecture: String = AIModel.deviceArchitectureName) throws -> URL {
        let dir = directory.resolvingSymlinksInPath()
        let items = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        let graphs = items.filter { $0 == "\(name).aimodel" || $0 == "\(name).aimodelc"
            || ($0.hasPrefix("\(name).") && $0.hasSuffix(".aimodelc") && $0.split(separator: ".").count == 3) }
            .sorted().map { dir.appendingPathComponent($0) }
        let archs = graphs.map(compiledArchitectures(of:))
        if let i = archs.firstIndex(where: { $0.contains(deviceArchitecture) }) { return graphs[i] }
        if let i = graphs.indices.first(where: { graphs[$0].pathExtension == "aimodel" && archs[$0].isEmpty }) {
            return graphs[i]
        }
        let found = Set(archs.joined())
        guard !found.isEmpty else { throw GateError.contract("assets: no \(name).aimodelc or \(name).aimodel in \(dir.path)") }
        throw GateError.contract("assets: \(name) is compiled for \(found.sorted().joined(separator: ", ")); this device is "
            + "\(deviceArchitecture) and there is no \(name).aimodel to specialize")
    }

    /// The `<arch>` of each `<function>-<arch>.mlirb` inside a compiled graph, else the one its name carries.
    static func compiledArchitectures(of graph: URL) -> [String] {
        let items = (try? FileManager.default.contentsOfDirectory(atPath: graph.path)) ?? []
        let inside = Set(items.compactMap { item -> String? in
            guard item.hasSuffix(".mlirb"), let dash = item.lastIndex(of: "-") else { return nil }
            let arch = String(item[item.index(after: dash)...].dropLast(".mlirb".count))
            return isArchitecture(arch) ? arch : nil
        })
        if !inside.isEmpty { return inside.sorted() }
        let named = graph.deletingPathExtension().pathExtension
        return graph.pathExtension == "aimodelc" && isArchitecture(named) ? [named] : []
    }

    static func isArchitecture(_ s: String) -> Bool {
        guard s.count >= 3, s.first == "h", let last = s.last, last.isLowercase, last.isASCII else { return false }
        return s.dropFirst().dropLast().allSatisfy { $0.isASCII && $0.isNumber }
    }
}

/// A compute unit by name: default (Core AI places the graph), gpu, ane, cpu. tnl/runtime.options.
enum Compute {
    static func options(_ name: String) -> SpecializationOptions {
        switch name {
        case "cpu": return .cpuOnly
        case "gpu": return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case "ane": return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        default: return .default
        }
    }
}

func secondsSince(_ t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

func jsonLine(_ obj: [String: Any]) -> String {
    let d = (try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
    return String(decoding: d, as: UTF8.self)
}
