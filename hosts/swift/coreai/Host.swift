// Host — one loaded titanet.aimodel and the seven-pass windowed scan that rebuilds TitaNet's three global reductions
// (five SqueezeExcite gates, pooling mean/std, attention softmax) exactly, from one 512-frame window swept over an
// utterance of any length. python/titanet_coreai.py is the same algorithm in Python. Host accumulation is Double.

import CoreAI
import Foundation

enum TitaNetError: Error, CustomStringConvertible {
    case contract(String)
    case audio(String)

    var description: String {
        switch self {
        case .contract(let s): return s
        case .audio(let s): return s
        }
    }
}

/// The fields of export.json the host needs.
struct ExportRecord: Decodable, Sendable {
    struct Signature: Decodable, Sendable { var inputs: [String]; var outputs: [String] }
    var shape: Int
    var margin: Int?
    var entrypoints: [String]
    var compute_dtype: String
    var seconds_of_audio: Double?
    var arm: String?
    var signature: [String: Signature]?
}

/// [(start, keep_lo, keep_hi)] in global frame coordinates: interiors tile [0, n) exactly once; each window carries a 46-frame margin of real context.
func windows(_ n: Int, _ F: Int, _ margin: Int) throws -> [(s: Int, lo: Int, hi: Int)] {
    if n <= F { return [(0, 0, n)] }
    let step = F - 2 * margin
    guard step > 0 else { throw TitaNetError.contract("window \(F) is too small for a \(margin)-frame margin on each side") }
    var out: [(Int, Int, Int)] = []
    var s = 0
    while true {
        let end = s + F
        out.append((s, s == 0 ? 0 : s + margin, end >= n ? n : end - margin))
        if end >= n { return out }
        s += step
    }
}

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

    init(graph: URL, record: ExportRecord, options: SpecializationOptions) async throws {
        // Everything is computed into locals first: a nested function or closure that touched `self` here would
        // capture it before all of its members were initialized, which Swift 6 refuses.
        let F = record.shape
        let m = try await AIModel(contentsOf: graph, options: options)
        let names = Set(m.functionNames)
        let hasScan = record.entrypoints.contains("scan") && names.contains("scan")
        let hasOnce = record.entrypoints.contains("once") && names.contains("once")
        guard hasScan || hasOnce else {
            throw TitaNetError.contract("\(graph.lastPathComponent): neither scan nor once; functions \(names.sorted())")
        }
        let scanD = hasScan ? try Host.descriptors(m, "scan") : [:]
        let onceD = hasOnce ? try Host.descriptors(m, "once") : [:]
        let headD = hasScan ? try Host.descriptors(m, "head") : [:]

        guard let melD = scanD["mel"] ?? onceD["mel"], melD.shape.count == 3, melD.shape[2] == F else {
            throw TitaNetError.contract("mel input is \((scanD["mel"] ?? onceD["mel"])?.shape.description ?? "missing"), "
                + "expected [1, mels, \(F)]")
        }
        var c = 0
        var se: [Int] = []
        if hasScan {
            guard let mean = scanD["mean"], mean.shape.count == 2 else { throw TitaNetError.contract("scan.mean is missing") }
            c = mean.shape[1]
            for j in 0..<5 {
                guard let d = scanD["s\(j)"], d.shape.count == 2 else { throw TitaNetError.contract("scan.s\(j) is missing") }
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
        guard let md = model.functionDescriptor(for: name) else { throw TitaNetError.contract("\(name): no function descriptor") }
        var out: [String: NDArrayDescriptor] = [:]
        for i in md.inputNames {
            guard let d = ND.descriptor(md.inputDescriptor(of: i)) else {
                throw TitaNetError.contract("\(name).\(i) is not an NDArray input")
            }
            out[i] = d
        }
        return out
    }

    private static func load(_ model: AIModel, _ name: String) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: name) else { throw TitaNetError.contract("\(name) did not load") }
        return fn
    }

    /// mel: [nMels * n] feature-major with the utterance's n valid frames first. emb [192] is returned unchecked.
    func embed(mel: [Float], n: Int, progress: @MainActor @Sendable @escaping (String) -> Void) async throws
        -> (emb: [Double], calls: Int, windows: Int) {
        guard n >= 1 else { throw TitaNetError.contract("no valid mel frame") }
        if n <= F, let once {
            let (mw, vw, _) = window(mel, n: n, s: 0, keep: (0, n))
            var inputs: [String: NDArray] = [:]
            inputs["mel"] = nd(mw, onceIn["mel"]!)
            inputs["valid"] = nd(vw, onceIn["valid"]!)
            var o = try await once.run(inputs: inputs)
            guard let e = o.remove("emb")?.ndArray else { throw TitaNetError.contract("once: no emb output") }
            return (ND.doubles(e), 1, 1)
        }
        guard let scan, let head else { throw TitaNetError.contract("\(n) frames need the scan path, which this bundle lacks") }
        let wins = try windows(n, F, margin)
        let nInv = 1.0 / Double(n)
        var calls = 0
        var sums = seSizes.map { [Double](repeating: 0, count: $0) }
        let zeroMean = [Double](repeating: 0, count: cOut)
        let unitStd = [Double](repeating: 1, count: cOut)

        // 1. the SqueezeExcite gate fixed point: five sweeps, the sums fed back as s0..s4
        for sweep in 0..<5 {
            await progress("Gate sweep \(sweep + 1) of 5")
            var new = seSizes.map { [Double](repeating: 0, count: $0) }
            for w in wins {
                let o = try await call(scan, mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: zeroMean, std: unitStd)
                calls += 1
                for j in 0..<5 {
                    guard let se = o["se\(j)"] else { throw TitaNetError.contract("scan: no se\(j) output") }
                    for k in 0..<min(new[j].count, se.count) { new[j][k] += se[k] }
                }
            }
            sums = new
        }

        // 2. the pooling mean / std of the encoder output
        await progress("Pooling statistics")
        var sx = [Double](repeating: 0, count: cOut)
        var sxx = [Double](repeating: 0, count: cOut)
        for w in wins {
            let o = try await call(scan, mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: zeroMean, std: unitStd)
            calls += 1
            guard let a = o["x_sum"], let b = o["x_sqsum"] else { throw TitaNetError.contract("scan: no x_sum / x_sqsum") }
            for k in 0..<cOut { sx[k] += a[k]; sxx[k] += b[k] }
        }
        var pmean = [Double](repeating: 0, count: cOut)
        var pstd = [Double](repeating: 0, count: cOut)
        for k in 0..<cOut {
            pmean[k] = sx[k] / Double(n)
            pstd[k] = (max(sxx[k] / Double(n) - pmean[k] * pmean[k], Host.stdEps)).squareRoot()
        }

        // 3. the attention softmax over time, combined across windows by running maximum
        await progress("Attentive pooling")
        var mx: [Double]? = nil
        var den = [Double](repeating: 0, count: cOut)
        var ax = [Double](repeating: 0, count: cOut)
        var axx = [Double](repeating: 0, count: cOut)
        for w in wins {
            let o = try await call(scan, mel: mel, n: n, w: w, nInv: nInv, sums: sums, mean: pmean, std: pstd)
            calls += 1
            guard let lmax = o["lmax"], let d = o["den"], let a = o["ax"], let aa = o["axx"] else {
                throw TitaNetError.contract("scan: no lmax / den / ax / axx")
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

        await progress("Embedding head")
        var inputs: [String: NDArray] = [:]
        inputs["mu"] = nd(mu.map { Float($0) }, headIn["mu"]!)
        inputs["sg"] = nd(sg.map { Float($0) }, headIn["sg"]!)
        var o = try await head.run(inputs: inputs)
        calls += 1
        guard let e = o.remove("emb")?.ndArray else { throw TitaNetError.contract("head: no emb output") }
        return (ND.doubles(e), calls, wins.count)
    }

    /// One `scan` call with its eleven inputs.
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

    /// mel [nMels, n] -> (mel_w [1, nMels, F], valid_w [1, 1, F], keep_w [1, 1, F]).
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

// MARK: - NDArray plumbing

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
