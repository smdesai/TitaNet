// MLGate — LeanGate for the Core ML export: the same headless device gate, with the same seven-pass host algorithm
// (tnl/runtime.py) calling Core ML `.mlmodelc` models instead of a Core AI bundle. LeanGate itself is left untouched
// so the Core AI measurements stay reproducible; status.json carries the same keys so the two compare directly.
//
// Library/Application Support/MLAssets/ (pushed with device.sh push):
//   bundles/<name>/{TitaNetScan.mlmodelc, TitaNetHead.mlmodelc, export.json}  as export/export_coreml.py writes them,
//                                                                           compiled on the Mac (xcrun coremlcompiler)
//   audio/{manifest.jsonl, mel/, emb/}                                       the accuracy fixtures, as for LeanGate
//   gate.json: {"tag": "...", "bundle": "bundles/<name>", "compute": "cpu_only|cpu_and_gpu|cpu_and_ne|all",
//               "manifest": "audio/manifest.jsonl"}   or   {..., "frames": [408, 828, 2052, 6472], "iters": 3}
// Documents/titanet_gate/ (pulled with device.sh run): status.json (phase, footprint, accuracy) and accuracy.jsonl.

import CoreML
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
                    Text("MLGate").font(.headline)
                    Text(gate.line).font(.system(.footnote, design: .monospaced))
                }
                .padding()
            }
            .task {
                UIApplication.shared.isIdleTimerDisabled = true
                await gate.run()
            }
        }
    }
}

struct GateConfig: Decodable {
    var tag: String?
    var bundle: String?
    var compute: String?
    var frames: [Int]?
    var iters: Int?
    var manifest: String?
    var limit: Int?
}

struct ManifestRow: Decodable {
    var id: String
    var mel: String
    var mel_len: Int
    var frames: Int
    var emb: String
    var seconds: Double
}

/// export/export_coreml.py's export.json, with the model files renamed to their compiled `.mlmodelc`.
struct ExportRecord: Decodable {
    struct Model: Decodable { var file: String; var inputs: [String]; var outputs: [String] }
    var precision: String
    var shape: Int
    var margin: Int
    var models: [String: Model]
    var moment_scale: Double?      // fp16c: the attention moments come back centred on `mean` and scaled by this
}

enum GateError: Error, CustomStringConvertible {
    case contract(String)
    var description: String { switch self { case .contract(let s): return s } }
}

// MARK: - the run (LeanGate's, with the Core ML host)

@MainActor @Observable
final class GateRun {
    var line = "starting"
    @ObservationIgnored private var status: [(String, Any)] = []
    @ObservationIgnored private var peakMB = 0.0
    @ObservationIgnored private let fm = FileManager.default
    @ObservationIgnored private let assets = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("MLAssets")
    @ObservationIgnored private let out = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("titanet_gate")

    func run() async {
        do {
            try fm.createDirectory(at: out, withIntermediateDirectories: true)
            let cfg = try JSONDecoder().decode(GateConfig.self, from: Data(contentsOf: assets.appendingPathComponent("gate.json")))
            let compute = cfg.compute ?? "cpu_and_gpu"
            let dir = assets.appendingPathComponent(cfg.bundle ?? "bundles/c_scan_f512_fp32")
            let tag = cfg.tag ?? dir.lastPathComponent
            let frames = cfg.frames ?? [408, 828, 2052, 6472]
            let iters = max(1, cfg.iters ?? 3)
            let rec = try JSONDecoder().decode(ExportRecord.self, from: Data(contentsOf: dir.appendingPathComponent("export.json")))

            note("phase", "load", ["tag": tag, "bundle": cfg.bundle ?? dir.lastPathComponent, "compute": compute,
                                   "runtime": "coreml", "precision": rec.precision, "shape": rec.shape,
                                   "margin": rec.margin, "frames": frames, "iters": iters])
            let launchMB = Self.footprintMB()
            mark(launchMB)

            let t0 = ContinuousClock.now
            let host = try await MLHost(bundle: dir, record: rec, compute: compute)
            let loadS = secondsSince(t0)
            let loadedMB = Self.footprintMB()
            mark(loadedMB)
            note("phase", "loaded", ["launch_mb": launchMB, "loaded_mb": loadedMB, "load_s": loadS])

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

    private func accuracy(_ host: MLHost, manifest: URL, limit: Int?) async throws {
        let dir = manifest.deletingLastPathComponent()
        let text = try String(contentsOf: manifest, encoding: .utf8)
        var rows: [ManifestRow] = []
        for line in text.split(separator: "\n") where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            rows.append(try JSONDecoder().decode(ManifestRow.self, from: Data(line.utf8)))
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
            let n = row.mel_len
            let raw = try Self.float32s(at: dir.appendingPathComponent(row.mel))
            guard raw.count == host.nMels * row.frames, n >= 1, n <= row.frames else {
                throw GateError.contract("\(row.id): mel has \(raw.count) floats for \(row.frames) frames, mel_len \(n)")
            }
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
                throw GateError.contract("\(row.id): reference has \(reference.count) values, the model returned \(r.emb.count)")
            }
            let finite = r.emb.allSatisfy { $0.isFinite }
            let cos = Self.cosine(r.emb, reference.map { Double($0) })
            if !finite || !cos.isFinite { allFinite = false }
            cosines.append(cos)
            latencies.append(latency)
            callsMax = max(callsMax, r.calls)
            lines.append(jsonLine(["id": row.id, "seconds": row.seconds, "n": n, "path": "scan", "calls": r.calls,
                                   "cos": cos, "latency_s": latency, "finite": finite]))
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

// MARK: - the host algorithm over Core ML

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

/// The two Core ML models and the seven-pass scan of tnl/runtime.py. Host accumulation is Double.
final class MLHost {
    static let stdEps = 1e-10
    static let seSizes = [1024, 1024, 1024, 1024, 3072]
    static let cOut = 3072
    let F: Int
    let margin: Int
    let momentScale: Double
    let centredExport: Bool
    let nMels = 80
    private let scan: MLModel
    private let head: MLModel

    init(bundle: URL, record: ExportRecord, compute: String) async throws {
        guard let s = record.models["scan"], let h = record.models["head"] else {
            throw GateError.contract("export.json: no scan / head model")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = switch compute {
        case "cpu_only": .cpuOnly
        case "cpu_and_ne": .cpuAndNeuralEngine
        case "all": .all
        default: .cpuAndGPU
        }
        scan = try await MLModel.load(contentsOf: bundle.appendingPathComponent(s.file), configuration: cfg)
        head = try await MLModel.load(contentsOf: bundle.appendingPathComponent(h.file), configuration: cfg)
        F = record.shape
        margin = record.margin
        momentScale = record.moment_scale ?? 1
        centredExport = record.moment_scale != nil
    }

    func embed(mel: [Float], n: Int, progress: @MainActor @escaping (String) -> Void) async throws
        -> (emb: [Double], calls: Int, windows: Int) {
        guard n >= 1 else { throw GateError.contract("no valid mel frame") }
        let wins = try windows(n, F, margin)
        let nInv = 1.0 / Double(n)
        let C = Self.cOut
        var calls = 0
        var sums = Self.seSizes.map { [Double](repeating: 0, count: $0) }
        let zeroMean = [Double](repeating: 0, count: C)
        let unitStd = [Double](repeating: 1, count: C)

        for sweep in 0..<5 {
            await progress("gate sweep \(sweep + 1)/5")
            var new = Self.seSizes.map { [Double](repeating: 0, count: $0) }
            for w in wins {
                let o = try await call(mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: zeroMean, std: unitStd)
                calls += 1
                for j in 0..<5 {
                    guard let se = o["se\(j)"] else { throw GateError.contract("scan: no se\(j) output") }
                    for k in 0..<min(new[j].count, se.count) { new[j][k] += se[k] }
                }
            }
            sums = new
        }

        await progress("pool pass")
        var sx = [Double](repeating: 0, count: C)
        var sxx = [Double](repeating: 0, count: C)
        for w in wins {
            let o = try await call(mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: zeroMean, std: unitStd)
            calls += 1
            guard let a = o["x_sum"], let b = o["x_sqsum"] else { throw GateError.contract("scan: no x_sum / x_sqsum") }
            for k in 0..<C { sx[k] += a[k]; sxx[k] += b[k] }
        }
        var pmean = [Double](repeating: 0, count: C)
        var pstd = [Double](repeating: 0, count: C)
        for k in 0..<C {
            pmean[k] = sx[k] / Double(n)
            pstd[k] = (max(sxx[k] / Double(n) - pmean[k] * pmean[k], Self.stdEps)).squareRoot()
        }

        // attention moments: plain (ax, axx), or from an fp16c export centred on the pooling mean and scaled by S
        // (ax_c, axx_c). The mean goes in already rounded to fp16, so the graph subtracts exactly what is added back.
        await progress("attention pass")
        let pm16 = centredExport ? pmean.map { Double(Float16($0)) } : pmean
        let S = momentScale
        var centred = false
        var mx: [Double]? = nil
        var den = [Double](repeating: 0, count: C)
        var ax = [Double](repeating: 0, count: C)
        var axx = [Double](repeating: 0, count: C)
        for w in wins {
            let o = try await call(mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: pm16, std: pstd)
            calls += 1
            centred = o["ax_c"] != nil
            guard let lmax = o["lmax"], let d = o["den"], let a = o[centred ? "ax_c" : "ax"],
                  let aa = o[centred ? "axx_c" : "axx"] else {
                throw GateError.contract("scan: no lmax / den / ax / axx (or ax_c / axx_c)")
            }
            var scale = [Double](repeating: 0, count: C)
            var f = [Double](repeating: 1, count: C)
            if let m = mx {
                var mNew = [Double](repeating: 0, count: C)
                for k in 0..<C {
                    mNew[k] = max(m[k], lmax[k])
                    scale[k] = exp(m[k] - mNew[k])
                    f[k] = exp(lmax[k] - mNew[k])
                }
                mx = mNew
            } else {
                mx = lmax
            }
            for k in 0..<C {
                den[k] = den[k] * scale[k] + d[k] * f[k]
                ax[k] = ax[k] * scale[k] + a[k] * f[k]
                axx[k] = axx[k] * scale[k] + aa[k] * f[k]
            }
        }
        var mu = [Float](repeating: 0, count: C)
        var sg = [Float](repeating: 0, count: C)
        for k in 0..<C {
            let s = centred ? S : 1
            let q = den[k] == 0 ? 0 : ax[k] / (s * den[k])
            mu[k] = Float(centred ? pm16[k] + q : q)
            let v = den[k] == 0 ? Self.stdEps : axx[k] / (s * s * den[k]) - q * q
            sg[k] = Float(max(v, Self.stdEps).squareRoot())
        }

        await progress("head")
        let feats = try MLDictionaryFeatureProvider(dictionary: [
            "mu": MLFeatureValue(multiArray: try Self.array(mu, [1, C])),
            "sg": MLFeatureValue(multiArray: try Self.array(sg, [1, C])),
        ])
        nonisolated(unsafe) let model = head     // MLModel predictions are thread-safe (Core ML docs)
        let o = try await model.prediction(from: feats)
        calls += 1
        guard let e = o.featureValue(for: "emb")?.multiArrayValue else { throw GateError.contract("head: no emb output") }
        return (Self.doubles(e), calls, wins.count)
    }

    /// One `scan` prediction with its eleven inputs, awaited, then a yield so the main actor (and the status file)
    /// keeps moving between calls.
    private func call(mel: [Float], n: Int, w: (s: Int, lo: Int, hi: Int), nInv: Double,
                      sums: [[Double]], mean: [Double], std: [Double]) async throws -> [String: [Double]] {
        var mw = [Float](repeating: 0, count: nMels * F)
        let take = max(0, min(F, n - w.s))
        if take > 0 {
            for c in 0..<nMels {
                let src = c * n + w.s
                for f in 0..<take { mw[c * F + f] = mel[src + f] }
            }
        }
        var vw = [Float](repeating: 0, count: F)
        for f in 0..<take { vw[f] = 1 }
        var kw = [Float](repeating: 0, count: F)
        let lo = max(0, w.lo - w.s), hi = min(F, w.hi - w.s)
        if hi > lo { for f in lo..<hi { kw[f] = 1 } }

        var d: [String: MLFeatureValue] = [
            "mel": MLFeatureValue(multiArray: try Self.array(mw, [1, nMels, F])),
            "valid": MLFeatureValue(multiArray: try Self.array(vw, [1, 1, F])),
            "keep": MLFeatureValue(multiArray: try Self.array(kw, [1, 1, F])),
            "n_inv": MLFeatureValue(multiArray: try Self.array([Float(nInv)], [1, 1])),
            "mean": MLFeatureValue(multiArray: try Self.array(mean.map { Float($0) }, [1, Self.cOut])),
            "std": MLFeatureValue(multiArray: try Self.array(std.map { Float($0) }, [1, Self.cOut])),
        ]
        for j in 0..<5 {
            d["s\(j)"] = MLFeatureValue(multiArray: try Self.array(sums[j].map { Float($0) }, [1, sums[j].count]))
        }
        nonisolated(unsafe) let model = scan     // MLModel predictions are thread-safe (Core ML docs)
        let o = try await model.prediction(from: try MLDictionaryFeatureProvider(dictionary: d))
        await Task.yield()
        var r: [String: [Double]] = [:]
        for name in o.featureNames {
            if let a = o.featureValue(for: name)?.multiArrayValue { r[name] = Self.doubles(a) }
        }
        return r
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
            return a.withUnsafeBufferPointer(ofType: Float16.self) { p in (0..<n).map { Double(p[$0 * s]) } }
        case .double:
            return a.withUnsafeBufferPointer(ofType: Double.self) { p in (0..<n).map { p[$0 * s] } }
        default:
            return a.withUnsafeBufferPointer(ofType: Float.self) { p in (0..<n).map { Double(p[$0 * s]) } }
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
