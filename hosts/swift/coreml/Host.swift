// Host — the seven-pass windowed scan that rebuilds TitaNet's three global reductions (five SqueezeExcite gates,
// pooling mean/std, attention softmax) exactly, from one 512-frame window swept over an utterance of any length.
// Written over `WindowedModel` (one `scan` call and the `head`); CoreMLModel.swift implements it. python/
// titanet_coreml.py is the same algorithm in Python. Host accumulation is Double.

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

/// [(start, keep_lo, keep_hi)] in global frame coordinates: interiors tile [0, n) exactly once.; each window carries a 46-frame margin of real context.
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

let stdEps = 1e-10

/// mel: [nMels * n] feature-major with the utterance's n valid frames first. emb [192] is returned unchecked.
func windowedEmbed(_ m: some WindowedModel, mel: [Float], n: Int, progress: @MainActor @Sendable @escaping (String) -> Void)
    async throws -> (emb: [Double], calls: Int, windows: Int) {
    guard n >= 1 else { throw TitaNetError.contract("no valid mel frame") }
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
                guard let se = o["se\(j)"] else { throw TitaNetError.contract("scan: no se\(j) output") }
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
        guard let a = o["x_sum"], let b = o["x_sqsum"] else { throw TitaNetError.contract("scan: no x_sum / x_sqsum") }
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
            throw TitaNetError.contract(S == nil ? "scan: no lmax / den / ax / axx" : "scan: no lmax / den / ax_c / axx_c")
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

/// mel [nMels, n] -> (mel_w [1, nMels, F], valid_w [1, 1, F], keep_w [1, 1, F]).
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
