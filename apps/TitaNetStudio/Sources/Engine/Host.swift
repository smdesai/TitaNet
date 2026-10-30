// Host — the seven-pass windowed scan of tnl/runtime.py, written once over a `WindowedModel` (one 512-frame `scan`
// call and the `head`), with a Core AI implementation here and a Core ML one in CoreMLModel.swift. The algorithm and
// the Core AI calls are LeanGate's (apps/LeanGate/Sources/GateApp.swift), which the device gates measured; the Core
// ML calls are MLGate's. Host accumulation is Double.

import CoreAI
import Foundation

enum StudioError: Error, CustomStringConvertible {
    case contract(String)
    case audio(String)

    var description: String {
        switch self {
        case .contract(let s): return s
        case .audio(let s): return s
        }
    }
}

/// One window's worth of the exported graph, whatever runtime executes it.
protocol WindowedModel: AnyObject {
    var F: Int { get }
    var margin: Int { get }
    var nMels: Int { get }
    var seSizes: [Int] { get }
    var cOut: Int { get }
    /// Set when `scan` returns its attention moments centred on `mean` and scaled by this power of two, as `ax_c`
    /// and `axx_c` (the fp16c Core ML export); nil when it returns plain `ax` and `axx`.
    var momentScale: Double? { get }
    /// `scan` with its eleven inputs, row-major: mel [1, nMels, F], valid / keep [1, 1, F], n_inv [1, 1],
    /// s0..s4 [1, seSizes[j]], mean / std [1, cOut]. Returns every output by name.
    func scan(mel: [Float], valid: [Float], keep: [Float], nInv: Float, sums: [[Float]], mean: [Float], std: [Float])
        async throws -> [String: [Double]]
    /// `head`: mu, sg [1, cOut] -> emb [1, 192].
    func head(mu: [Float], sg: [Float]) async throws -> [Double]
}

/// [(start, keep_lo, keep_hi)] in global frame coordinates: interiors tile [0, n) exactly once. tnl/stream.windows.
func windows(_ n: Int, _ F: Int, _ margin: Int) throws -> [(s: Int, lo: Int, hi: Int)] {
    if n <= F { return [(0, 0, n)] }
    let step = F - 2 * margin
    guard step > 0 else { throw StudioError.contract("window \(F) is too small for a \(margin)-frame margin on each side") }
    var out: [(Int, Int, Int)] = []
    var s = 0
    while true {
        let end = s + F
        out.append((s, s == 0 ? 0 : s + margin, end >= n ? n : end - margin))
        if end >= n { return out }
        s += step
    }
}

let stdEps = 1e-10

/// mel: [nMels * n] feature-major with the utterance's n valid frames first. emb [192] is returned unchecked.
func windowedEmbed(_ m: some WindowedModel, mel: [Float], n: Int, progress: @MainActor @Sendable @escaping (String) -> Void)
    async throws -> (emb: [Double], calls: Int, windows: Int) {
    guard n >= 1 else { throw StudioError.contract("no valid mel frame") }
    let wins = try windows(n, m.F, m.margin)
    let nInv = Float(1.0 / Double(n))
    let C = m.cOut
    var calls = 0
    var sums = m.seSizes.map { [Double](repeating: 0, count: $0) }
    let zeroMean = [Float](repeating: 0, count: C)
    let unitStd = [Float](repeating: 1, count: C)

    func call(_ w: (s: Int, lo: Int, hi: Int), _ mean: [Float], _ std: [Float]) async throws -> [String: [Double]] {
        let (mw, vw, kw) = window(mel, n: n, F: m.F, nMels: m.nMels, s: w.s, keep: (w.lo, w.hi))
        calls += 1
        return try await m.scan(mel: mw, valid: vw, keep: kw, nInv: nInv, sums: sums.map { $0.map { Float($0) } },
                                mean: mean, std: std)
    }

    // 1. the SqueezeExcite gate fixed point: five sweeps, the sums fed back as s0..s4
    for sweep in 0..<5 {
        await progress("Gate sweep \(sweep + 1) of 5")
        var new = m.seSizes.map { [Double](repeating: 0, count: $0) }
        for w in wins {
            let o = try await call(w, zeroMean, unitStd)
            for j in 0..<5 {
                guard let se = o["se\(j)"] else { throw StudioError.contract("scan: no se\(j) output") }
                for k in 0..<min(new[j].count, se.count) { new[j][k] += se[k] }
            }
        }
        sums = new
    }

    // 2. the pooling mean / std of the encoder output
    await progress("Pooling statistics")
    var sx = [Double](repeating: 0, count: C)
    var sxx = [Double](repeating: 0, count: C)
    for w in wins {
        let o = try await call(w, zeroMean, unitStd)
        guard let a = o["x_sum"], let b = o["x_sqsum"] else { throw StudioError.contract("scan: no x_sum / x_sqsum") }
        for k in 0..<C { sx[k] += a[k]; sxx[k] += b[k] }
    }
    var pmean = [Float](repeating: 0, count: C)
    var pstd = [Float](repeating: 0, count: C)
    for k in 0..<C {
        let mu = sx[k] / Double(n)
        pmean[k] = Float(mu)
        pstd[k] = Float((max(sxx[k] / Double(n) - mu * mu, stdEps)).squareRoot())
    }

    // 3. the attention softmax over time, combined across windows by running maximum. With centred moments the mean
    //    goes in already rounded to fp16, so the graph subtracts exactly the value added back below.
    await progress("Attentive pooling")
    let S = m.momentScale
    if S != nil { pmean = pmean.map { Float(Float16($0)) } }
    var mx: [Double]? = nil
    var den = [Double](repeating: 0, count: C)
    var ax = [Double](repeating: 0, count: C)
    var axx = [Double](repeating: 0, count: C)
    for w in wins {
        let o = try await call(w, pmean, pstd)
        guard let lmax = o["lmax"], let d = o["den"], let a = o[S == nil ? "ax" : "ax_c"],
              let aa = o[S == nil ? "axx" : "axx_c"] else {
            throw StudioError.contract(S == nil ? "scan: no lmax / den / ax / axx" : "scan: no lmax / den / ax_c / axx_c")
        }
        var scale = [Double](repeating: 0, count: C)
        var f = [Double](repeating: 1, count: C)
        if let mPrev = mx {
            var mNew = [Double](repeating: 0, count: C)
            for k in 0..<C {
                mNew[k] = max(mPrev[k], lmax[k])
                scale[k] = exp(mPrev[k] - mNew[k])
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
    let s = S ?? 1
    for k in 0..<C {
        let q = den[k] == 0 ? 0 : ax[k] / (s * den[k])
        mu[k] = Float(S == nil ? q : Double(pmean[k]) + q)
        let v = den[k] == 0 ? stdEps : axx[k] / (s * s * den[k]) - q * q
        sg[k] = Float(max(v, stdEps).squareRoot())
    }

    await progress("Embedding head")
    let emb = try await m.head(mu: mu, sg: sg)
    return (emb, calls + 1, wins.count)
}

/// mel [nMels, n] -> (mel_w [1, nMels, F], valid_w [1, 1, F], keep_w [1, 1, F]). tnl/runtime._window.
private func window(_ mel: [Float], n: Int, F: Int, nMels: Int, s: Int, keep: (Int, Int)) -> ([Float], [Float], [Float]) {
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

// MARK: - Core AI

/// The fields of export.json this app needs (export/export_coreai.py writes the rest).
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

/// The titanet-lean Core AI bundle: `scan` and `head` in one titanet.aimodel.
final class CoreAIModel: WindowedModel {
    let record: ExportRecord
    let F: Int
    let margin: Int
    let nMels: Int
    let cOut: Int
    let seSizes: [Int]
    let momentScale: Double? = nil
    private let model: AIModel
    private let scanFn: InferenceFunction
    private let headFn: InferenceFunction
    private let scanIn: [String: NDArrayDescriptor]
    private let headIn: [String: NDArrayDescriptor]
    private let scanOutputs: [String]

    init(graph: URL, record: ExportRecord, options: SpecializationOptions) async throws {
        // Everything is computed into locals first: a nested function or closure that touched `self` here would
        // capture it before all of its members were initialized, which Swift 6 refuses.
        let F = record.shape
        let m = try await AIModel(contentsOf: graph, options: options)
        let names = Set(m.functionNames)
        guard record.entrypoints.contains("scan"), names.contains("scan"), names.contains("head") else {
            throw StudioError.contract("\(graph.lastPathComponent): needs scan and head; functions \(names.sorted())")
        }
        let scanD = try CoreAIModel.descriptors(m, "scan")
        let headD = try CoreAIModel.descriptors(m, "head")
        guard let melD = scanD["mel"], melD.shape.count == 3, melD.shape[2] == F else {
            throw StudioError.contract("mel input is \(scanD["mel"]?.shape.description ?? "missing"), expected [1, mels, \(F)]")
        }
        guard let mean = scanD["mean"], mean.shape.count == 2 else { throw StudioError.contract("scan.mean is missing") }
        var se: [Int] = []
        for j in 0..<5 {
            guard let d = scanD["s\(j)"], d.shape.count == 2 else { throw StudioError.contract("scan.s\(j) is missing") }
            se.append(d.shape[1])
        }

        self.record = record
        self.F = F
        self.margin = record.margin ?? 46
        self.model = m
        self.scanOutputs = record.signature?["scan"]?.outputs
            ?? ["se0", "se1", "se2", "se3", "se4", "x_sum", "x_sqsum", "lmax", "den", "ax", "axx"]
        self.scanIn = scanD
        self.headIn = headD
        self.scanFn = try CoreAIModel.load(m, "scan")
        self.headFn = try CoreAIModel.load(m, "head")
        self.nMels = melD.shape[1]
        self.cOut = mean.shape[1]
        self.seSizes = se
    }

    /// Every NDArray input of `name`, by name, as the graph itself declares it (shape and scalar type).
    private static func descriptors(_ model: AIModel, _ name: String) throws -> [String: NDArrayDescriptor] {
        guard let md = model.functionDescriptor(for: name) else { throw StudioError.contract("\(name): no function descriptor") }
        var out: [String: NDArrayDescriptor] = [:]
        for i in md.inputNames {
            guard let d = ND.descriptor(md.inputDescriptor(of: i)) else {
                throw StudioError.contract("\(name).\(i) is not an NDArray input")
            }
            out[i] = d
        }
        return out
    }

    private static func load(_ model: AIModel, _ name: String) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: name) else { throw StudioError.contract("\(name) did not load") }
        return fn
    }

    func scan(mel: [Float], valid: [Float], keep: [Float], nInv: Float, sums: [[Float]], mean: [Float], std: [Float])
        async throws -> [String: [Double]] {
        var inputs: [String: NDArray] = [:]
        inputs["mel"] = nd(mel, scanIn["mel"]!)
        inputs["valid"] = nd(valid, scanIn["valid"]!)
        inputs["keep"] = nd(keep, scanIn["keep"]!)
        inputs["n_inv"] = nd([nInv], scanIn["n_inv"]!)
        inputs["mean"] = nd(mean, scanIn["mean"]!)
        inputs["std"] = nd(std, scanIn["std"]!)
        for j in 0..<5 { inputs["s\(j)"] = nd(sums[j], scanIn["s\(j)"]!) }
        var out = try await scanFn.run(inputs: inputs)
        var r: [String: [Double]] = [:]
        for name in scanOutputs {
            guard let a = out.remove(name)?.ndArray else { continue }
            r[name] = ND.doubles(a)
        }
        return r
    }

    func head(mu: [Float], sg: [Float]) async throws -> [Double] {
        var inputs: [String: NDArray] = [:]
        inputs["mu"] = nd(mu, headIn["mu"]!)
        inputs["sg"] = nd(sg, headIn["sg"]!)
        var o = try await headFn.run(inputs: inputs)
        guard let e = o.remove("emb")?.ndArray else { throw StudioError.contract("head: no emb output") }
        return ND.doubles(e)
    }

    /// Input dtype comes from the graph's own descriptor, which export.json `compute_dtype` set.
    private func nd(_ v: [Float], _ d: NDArrayDescriptor) -> NDArray {
        d.scalarType == .float16 ? ND.make(v.map { Float16($0) }, d) : ND.make(v, d)
    }
}

// MARK: - plumbing lifted from TitaNetKit (ND.swift)

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
