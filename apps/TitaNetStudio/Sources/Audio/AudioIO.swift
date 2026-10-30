// AudioIO — every way audio enters or leaves the app, as 16 kHz mono Float32 (the model's only input rate).
//   AudioLoader  a file to samples. A 16 kHz mono PCM16 / float32 WAV is read exactly as soundfile reads it (the
//                gates' path); anything else AVFoundation can decode is downmixed and resampled with AVAudioConverter,
//                which NeMo's librosa path does not reproduce bit for bit.
//   MicCapture   the microphone to samples, resampled in the tap, with an RMS level every 50 ms for the live meter.
//   WAV          samples to an in-memory float32 WAV, for playback.

import AVFoundation
import Foundation
import os

enum AudioLoader {
    static let rate = Double(SpeakerEngine.sampleRate)

    static func load(_ url: URL) throws -> [Float] {
        if let exact = try? WAV.read16k(url) { return exact }
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch {
            throw StudioError.audio("\(url.lastPathComponent) is not an audio file this device can decode.")
        }
        let fmt = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames) else {
            throw StudioError.audio("\(url.lastPathComponent) has no audio.")
        }
        try file.read(into: buf)
        guard let ch = buf.floatChannelData else { throw StudioError.audio("\(url.lastPathComponent): unsupported sample format.") }
        // downmix by averaging channels
        let n = Int(buf.frameLength), channels = Int(fmt.channelCount)
        var mono = [Float](repeating: 0, count: n)
        for c in 0..<channels {
            let p = ch[c]
            for i in 0..<n { mono[i] += p[i] }
        }
        if channels > 1 { for i in 0..<n { mono[i] /= Float(channels) } }
        return fmt.sampleRate == rate ? mono : try resample(mono, from: fmt.sampleRate)
    }

    static func resample(_ x: [Float], from sr: Double) throws -> [Float] {
        guard let inFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sr, channels: 1, interleaved: false),
              let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(x.count)),
              let conv = AVAudioConverter(from: inFmt, to: MicCapture.target) else {
            throw StudioError.audio("Cannot resample \(Int(sr)) Hz audio.")
        }
        inBuf.frameLength = AVAudioFrameCount(x.count)
        x.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: x.count) }
        var out: [Float] = []
        out.reserveCapacity(Int(Double(x.count) * rate / sr) + 1024)
        let fed = Flag()
        while true {
            guard let chunk = AVAudioPCMBuffer(pcmFormat: MicCapture.target, frameCapacity: 16384) else { break }
            var error: NSError?
            let status = conv.convert(to: chunk, error: &error) { _, s in
                if fed.value { s.pointee = .endOfStream; return nil }
                fed.value = true
                s.pointee = .haveData
                return inBuf
            }
            if let error { throw StudioError.audio("Resampling failed: \(error.localizedDescription)") }
            if chunk.frameLength > 0 {
                out.append(contentsOf: UnsafeBufferPointer(start: chunk.floatChannelData![0], count: Int(chunk.frameLength)))
            }
            if status == .endOfStream || status == .error || (status == .inputRanDry && chunk.frameLength == 0) { break }
        }
        return out
    }
}

/// A mutable flag an AVAudioConverter input block can set (the block may be @Sendable).
private final class Flag: @unchecked Sendable { var value = false }

final class MicCapture: @unchecked Sendable {
    static let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: AudioLoader.rate, channels: 1,
                                      interleaved: false)!
    static let levelHop = 800        // 50 ms at 16 kHz

    private struct State {
        var samples: [Float] = []
        var levels: [Float] = []
        var pending: [Float] = []
    }

    private let engine = AVAudioEngine()
    private let state = OSAllocatedUnfairLock(initialState: State())
    private var converter: AVAudioConverter?

    func start() throws {
        state.withLock { $0 = State() }
        let input = engine.inputNode
        let fmt = input.outputFormat(forBus: 0)
        guard fmt.sampleRate > 0, fmt.channelCount > 0 else { throw StudioError.audio("No microphone is available.") }
        guard let conv = AVAudioConverter(from: fmt, to: Self.target) else {
            throw StudioError.audio("Cannot convert \(Int(fmt.sampleRate)) Hz microphone audio.")
        }
        converter = conv
        install(on: input, format: fmt)
        engine.prepare()
        try engine.start()
    }

    /// Kept out of any actor: the tap runs on the audio render thread.
    private nonisolated func install(on input: AVAudioInputNode, format: AVAudioFormat) {
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            self?.consume(buffer)
        }
    }

    private func consume(_ buffer: AVAudioPCMBuffer) {
        guard let conv = converter else { return }
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * AudioLoader.rate / buffer.format.sampleRate) + 256
        guard let out = AVAudioPCMBuffer(pcmFormat: Self.target, frameCapacity: cap) else { return }
        let fed = Flag()
        var error: NSError?
        conv.convert(to: out, error: &error) { _, s in
            if fed.value { s.pointee = .noDataNow; return nil }     // not endOfStream: the resampler keeps its state
            fed.value = true
            s.pointee = .haveData
            return buffer
        }
        guard error == nil, out.frameLength > 0, let p = out.floatChannelData?[0] else { return }
        let chunk = Array(UnsafeBufferPointer(start: p, count: Int(out.frameLength)))
        state.withLock { st in
            st.samples += chunk
            st.pending += chunk
            while st.pending.count >= Self.levelHop {
                var ss: Float = 0
                for i in 0..<Self.levelHop { ss += st.pending[i] * st.pending[i] }
                st.levels.append((ss / Float(Self.levelHop)).squareRoot())
                st.pending.removeFirst(Self.levelHop)
            }
        }
    }

    /// (levels so far, seconds captured)
    func snapshot(lastLevels k: Int) -> ([Float], Double) {
        state.withLock { (Array($0.levels.suffix(k)), Double($0.samples.count) / AudioLoader.rate) }
    }

    func stop() -> [Float] {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        return state.withLock { $0.samples }
    }
}

enum WAV {
    /// A 16 kHz mono RIFF/WAVE in PCM16 or IEEE float32, as soundfile.read(dtype="float32") reads it; nil otherwise.
    /// TitaNetKit's WAV.read, returning nil instead of refusing so the caller can fall back to AVFoundation.
    static func read16k(_ url: URL) throws -> [Float]? {
        let d = try Data(contentsOf: url, options: .mappedIfSafe)
        func u32(_ at: Int) -> UInt32 { UInt32(littleEndian: d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: at, as: UInt32.self) }) }
        func u16(_ at: Int) -> UInt16 { UInt16(littleEndian: d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: at, as: UInt16.self) }) }
        guard d.count >= 12, d.prefix(4) == Data("RIFF".utf8), d[8..<12] == Data("WAVE".utf8) else { return nil }
        var at = 12
        var format: (tag: UInt16, channels: UInt16, rate: UInt32, bits: UInt16)?
        while at + 8 <= d.count {
            let id = String(decoding: d[at..<(at + 4)], as: UTF8.self)
            let size = Int(u32(at + 4))
            let body = at + 8
            if id == "fmt " {
                guard size >= 16, body + 16 <= d.count else { return nil }
                var tag = u16(body)
                if tag == 0xFFFE, size >= 26, body + 26 <= d.count { tag = u16(body + 24) }
                format = (tag, u16(body + 2), u32(body + 4), u16(body + 14))
            } else if id == "data" {
                guard let f = format, f.channels == 1, f.rate == 16000 else { return nil }
                let n = min(size, d.count - body)
                switch (f.tag, f.bits) {
                case (1, 16):
                    return d.withUnsafeBytes { raw in
                        (0..<(n / 2)).map { Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: body + 2 * $0, as: Int16.self))) / 32768 }
                    }
                case (3, 32):
                    return d.withUnsafeBytes { raw in
                        (0..<(n / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: body + 4 * $0, as: UInt32.self))) }
                    }
                default:
                    return nil
                }
            }
            at = body + size + (size & 1)
        }
        return nil
    }

    /// A 16 kHz mono IEEE float32 WAV in memory.
    static func data(_ x: [Float]) -> Data {
        var d = Data()
        d.reserveCapacity(44 + 4 * x.count)
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let bytes = UInt32(x.count * 4)
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(3); u16(1); u32(16000); u32(64000); u16(4); u16(32)
        d.append(contentsOf: Array("data".utf8)); u32(bytes)
        x.withUnsafeBytes { d.append(contentsOf: $0) }
        return d
    }
}
