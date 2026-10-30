// MelFrontend — NeMo's FilterbankFeatures (eval mode) for one utterance, the front end TitaNet-Large was trained on.
// Needs frontend/mel_fb.f32 (slaney filterbank [80, 257]) and frontend/mel_window.f32 (Hann(400), periodic=False),
// both the checkpoint's own buffers. python/titanet_coreai.py implements the same steps in numpy.
//
//   samples Float [N] (16 kHz) -> mel Float [80, N / 160 + 1] row-major (feature-major), valid frames n = N / 160
//   1 preemphasis  y[0] = x[0], y[i] = x[i] - 0.97 x[i-1]                        (Float)
//   2 STFT         centred, 256 zeros each side, Hann(400, periodic=false) placed at offset 56 of 512; the DFT summed
//                  in Double (vDSP_mmulD over an exact (k j mod 512) cos / -sin basis); re / im rounded to Float
//   3 power        sqrt(re^2 + im^2) then squared, in Float                     (NeMo: magnitude, then pow(2))
//   4 mel          filterbank [80, 257] x power, summed in Double, rounded to Float
//   5 log          log(mel + 2^-24) in Double, rounded to Float
//   6 normalize    per feature over the n valid frames: reference-centred mean, unbiased std + 1e-5 (Float), frames
//                  >= n set to 0

import Accelerate
import Foundation

final class MelFrontend: @unchecked Sendable {
    static let nFFT = 512, hop = 160, win = 400, nMels = 80, bins = 257
    static let preemph: Float = 0.97
    static let logGuard: Double = Double(Float(pow(2.0, -24.0)))
    static let stdEps: Float = 1e-5

    private let window: [Double]      // [512], the 400 taps centred
    private let basis: [Double]       // [512, 514]: column k = cos(2 pi k j / 512), column 257 + k = -sin(...)
    private let fb: [Double]          // [80, 257]

    init(filterbank: [Float], window w: [Float]) {
        precondition(filterbank.count == Self.nMels * Self.bins && w.count == Self.win)
        var win = [Double](repeating: 0, count: Self.nFFT)
        let off = (Self.nFFT - Self.win) / 2
        for i in 0..<Self.win { win[off + i] = Double(w[i]) }
        window = win
        var b = [Double](repeating: 0, count: Self.nFFT * 2 * Self.bins)
        for j in 0..<Self.nFFT {
            for k in 0..<Self.bins {
                let ang = 2.0 * Double.pi * Double((k * j) % Self.nFFT) / Double(Self.nFFT)
                b[j * 2 * Self.bins + k] = cos(ang)
                b[j * 2 * Self.bins + Self.bins + k] = -sin(ang)
            }
        }
        basis = b
        fb = filterbank.map(Double.init)
    }

    /// The normalized log-mel of `samples`: (mel [80 * frames] feature-major, frames, valid frames).
    func callAsFunction(_ x: [Float]) -> (mel: [Float], frames: Int, valid: Int) {
        let n = x.count
        let valid = n / Self.hop
        guard n > 0 else { return ([Float](repeating: 0, count: Self.nMels), 1, 0) }
        // 1 preemphasis
        var y = [Float](repeating: 0, count: n)
        y[0] = x[0]
        for i in 1..<n { y[i] = x[i] - Self.preemph * x[i - 1] }
        // 2 framing of the centred, zero-padded signal
        let pad = Self.nFFT / 2
        let total = n + 2 * pad
        let frames = 1 + (total - Self.nFFT) / Self.hop
        var framed = [Double](repeating: 0, count: frames * Self.nFFT)
        for f in 0..<frames {
            let start = f * Self.hop - pad
            for j in 0..<Self.nFFT {
                let s = start + j
                if s >= 0 && s < n { framed[f * Self.nFFT + j] = Double(y[s]) * window[j] }
            }
        }
        var spec = [Double](repeating: 0, count: frames * 2 * Self.bins)
        vDSP_mmulD(framed, 1, basis, 1, &spec, 1, vDSP_Length(frames), vDSP_Length(2 * Self.bins), vDSP_Length(Self.nFFT))
        // 3 power, Float, stored [257, frames] in Double for the mel product
        var power = [Double](repeating: 0, count: Self.bins * frames)
        for f in 0..<frames {
            for k in 0..<Self.bins {
                let re = Float(spec[f * 2 * Self.bins + k]), im = Float(spec[f * 2 * Self.bins + Self.bins + k])
                let mag = (re * re + im * im).squareRoot()
                power[k * frames + f] = Double(mag * mag)
            }
        }
        // 4 mel, 5 log
        var mel64 = [Double](repeating: 0, count: Self.nMels * frames)
        vDSP_mmulD(fb, 1, power, 1, &mel64, 1, vDSP_Length(Self.nMels), vDSP_Length(frames), vDSP_Length(Self.bins))
        var mel = [Float](repeating: 0, count: Self.nMels * frames)
        for i in 0..<mel.count {
            mel[i] = Float(log(Double(Float(mel64[i])) + Self.logGuard))
        }
        // 6 normalize
        Self.normalize(&mel, frames: frames, valid: valid)
        return (mel, frames, valid)
    }

    /// normalize_batch(per_feature) for one utterance (model/host_mel.normalize).
    static func normalize(_ x: inout [Float], frames: Int, valid n: Int) {
        for c in 0..<nMels {
            let row = c * frames
            if n == 0 {
                for f in 0..<frames { x[row + f] = 0 }
                continue
            }
            let ref = Double(x[row])
            var sum = 0.0
            for f in 0..<n { sum += Double(Float(Double(x[row + f]) - ref)) }
            let mean = Float(ref + sum / Double(n))
            var ss = 0.0
            for f in 0..<n {
                let d = Double(Float(Double(x[row + f]) - Double(mean)))
                ss += d * d
            }
            let std: Float = (n > 1 ? Float((ss / (Double(n) - 1.0)).squareRoot()) : 0) + stdEps
            for f in 0..<n { x[row + f] = (x[row + f] - mean) / std }
            for f in n..<frames { x[row + f] = 0 }
        }
    }
}
