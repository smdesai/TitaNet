// ClipCard — one clip slot: record or import, then its waveform, playback, voiceprint and the cost of embedding it.

import SwiftUI
import UniformTypeIdentifiers

struct ClipCard: View {
    @Environment(AppModel.self) private var model
    let slot: ClipSlot
    var title: String
    var badge: String
    var tint: Color

    @State private var importing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            content
                .frame(maxWidth: .infinity)
                .transition(.opacity.combined(with: .scale(scale: 0.97)))
        }
        .animation(.snappy, value: stateKey)
        .card(tint: tint)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.audio]) { result in
            guard case .success(let url) = result else { return }
            Task { await model.importFile(url, into: slot) }
        }
    }

    private var stateKey: String {
        switch slot.state {
        case .empty: "empty"
        case .recording: "recording"
        case .processing: "processing"
        case .ready(let c): c.id.uuidString
        case .failed: "failed"
        }
    }

    // MARK: header

    private var header: some View {
        HStack(spacing: 12) {
            Text(badge)
                .font(.system(.headline, design: .rounded, weight: .bold))
                .foregroundStyle(.black.opacity(0.8))
                .frame(width: 32, height: 32)
                .background(tint.gradient, in: Circle())
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.headline).foregroundStyle(.white)
                if let clip = slot.clip {
                    Label(clip.source.label, systemImage: clip.source.symbol)
                        .font(.caption)
                        .lineLimit(1)
                        .foregroundStyle(.white.opacity(0.55))
                }
            }
            Spacer()
            if slot.clip != nil || isFailed {
                Menu {
                    Button("Record again", systemImage: "mic") { Task { await model.toggleRecording(slot) } }
                    Button("Choose another file", systemImage: "folder") { importing = true }
                    Divider()
                    Button("Clear", systemImage: "trash", role: .destructive) { model.clear(slot) }
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.body.weight(.semibold))
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
            }
        }
    }

    private var isFailed: Bool { if case .failed = slot.state { true } else { false } }

    // MARK: content

    @ViewBuilder private var content: some View {
        switch slot.state {
        case .empty:
            empty
        case .recording:
            recording
        case .processing(let phase):
            processing(phase)
        case .ready(let clip):
            ready(clip)
        case .failed(let message):
            failed(message)
        }
    }

    private var empty: some View {
        VStack(spacing: 14) {
            Image(systemName: "waveform")
                .font(.system(size: 30, weight: .light))
                .foregroundStyle(tint.opacity(0.7))
                .symbolEffect(.variableColor.iterative.dimInactiveLayers, options: .repeat(.continuous))
            HStack(spacing: 10) {
                Button { Task { await model.toggleRecording(slot) } } label: {
                    Label("Record", systemImage: "mic.fill").frame(maxWidth: .infinity)
                }
                .buttonStyle(.glassProminent)
                .tint(tint.opacity(0.85))
                Button { importing = true } label: {
                    Label("Import", systemImage: "square.and.arrow.down").frame(maxWidth: .infinity)
                }
                .buttonStyle(.glass)
            }
            .controlSize(.large)
            .fontWeight(.semibold)
        }
        .padding(.vertical, 6)
    }

    private var recording: some View {
        VStack(spacing: 14) {
            LiveWaveform(levels: model.recorder.levels, bars: Recorder.bars, tint: tint)
                .frame(height: 64)
            HStack {
                Circle().fill(Theme.record).frame(width: 9, height: 9)
                    .phaseAnimator([1.0, 0.25]) { v, o in v.opacity(o) } animation: { _ in .easeInOut(duration: 0.6) }
                Text(model.recorder.elapsed.clock)
                    .font(.system(.title3, design: .rounded, weight: .semibold))
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .foregroundStyle(.white)
                Text("/ \(Int(Recorder.limit)) s")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.45))
                Spacer()
                Button { Task { await model.stopRecording(slot) } } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.glassProminent)
                .tint(Theme.record)
                .fontWeight(.semibold)
            }
            Text(model.recorder.elapsed < 3 ? "Keep talking — 3 s or more gives a steadier voiceprint" : "Tap Stop when you're done")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.5))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func processing(_ phase: String) -> some View {
        VStack(spacing: 14) {
            WaveformView(envelope: slot.preview, tint: tint.opacity(0.45))
                .frame(height: 54)
                .overlay {
                    if slot.preview.isEmpty { ProgressView().tint(.white) }
                }
            HStack(spacing: 10) {
                ProgressView().controlSize(.small).tint(tint)
                Text(phase)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white.opacity(0.8))
                    .contentTransition(.opacity)
                Spacer()
            }
        }
    }

    private func ready(_ clip: Clip) -> some View {
        let playing = model.player.playing == clip.id
        return VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Button { model.player.toggle(clip) } label: {
                    Image(systemName: playing ? "pause.fill" : "play.fill")
                        .font(.body.weight(.bold))
                        .contentTransition(.symbolEffect(.replace))
                        .frame(width: 40, height: 40)
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .tint(tint.opacity(0.8))
                WaveformView(envelope: clip.envelope, tint: tint, progress: playing ? model.player.progress : nil)
                    .frame(height: 46)
                Text(clip.seconds.clock)
                    .font(.caption.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(.white.opacity(0.7))
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Voiceprint · \(clip.embedding.vector.count)-d")
                    .font(.caption2.weight(.semibold))
                    .textCase(.uppercase)
                    .kerning(0.8)
                    .foregroundStyle(.white.opacity(0.45))
                VoiceprintView(vector: clip.embedding.vector, tint: tint)
                    .frame(height: 38)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    let e = clip.embedding
                    if !e.engine.isEmpty { StatChip(symbol: "cpu", text: e.engine) }
                    StatChip(symbol: "bolt.fill", text: String(format: "%.0f ms", (e.featureSeconds + e.modelSeconds) * 1000))
                    StatChip(symbol: "rectangle.split.3x1", text: "\(e.windows) window\(e.windows == 1 ? "" : "s")")
                    StatChip(symbol: "arrow.triangle.2.circlepath", text: "\(e.calls) calls")
                    StatChip(symbol: "waveform.path", text: "\(e.frames) frames")
                    if clip.trimmed { StatChip(symbol: "scissors", text: "first \(Int(AppModel.maxSeconds)) s") }
                }
            }
        }
    }

    private func failed(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.subheadline)
                .foregroundStyle(Theme.mismatch)
            HStack(spacing: 10) {
                Button("Record", systemImage: "mic.fill") { Task { await model.toggleRecording(slot) } }
                Button("Import", systemImage: "square.and.arrow.down") { importing = true }
            }
            .buttonStyle(.glass)
        }
    }
}
