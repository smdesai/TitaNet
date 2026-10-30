// Components — the drawings: clip waveform, live meter, the 192-d voiceprint, the score gauge, chips and pills.

import SwiftUI

/// A clip's envelope as mirrored rounded bars; `progress` lights the bars already played.
struct WaveformView: View {
    var envelope: [Float]
    var tint: Color
    var progress: Double? = nil

    var body: some View {
        Canvas { ctx, size in
            guard !envelope.isEmpty else { return }
            let n = envelope.count
            let slot = size.width / CGFloat(n)
            let w = max(1.5, slot * 0.58)
            for (i, v) in envelope.enumerated() {
                let h = max(3, CGFloat(v.squareRoot()) * size.height * 0.95)
                let r = CGRect(x: CGFloat(i) * slot + (slot - w) / 2, y: (size.height - h) / 2, width: w, height: h)
                let played = progress.map { Double(i) / Double(n) < $0 } ?? true
                ctx.fill(Path(roundedRect: r, cornerRadius: w / 2),
                         with: .color(played ? tint : tint.opacity(0.35)))
            }
        }
    }
}

/// The microphone's last few seconds of RMS, newest on the right.
struct LiveWaveform: View {
    var levels: [Float]
    var bars: Int
    var tint: Color

    var body: some View {
        Canvas { ctx, size in
            let slot = size.width / CGFloat(bars)
            let w = max(2, slot * 0.55)
            let pad = bars - levels.count
            for i in 0..<bars {
                let v = i < pad ? 0 : min(1, CGFloat(levels[i - pad]).squareRoot() * 2.6)
                let h = max(4, v * size.height)
                let r = CGRect(x: CGFloat(i) * slot + (slot - w) / 2, y: (size.height - h) / 2, width: w, height: h)
                ctx.fill(Path(roundedRect: r, cornerRadius: w / 2), with: .color(tint.opacity(i < pad ? 0.25 : 1)))
            }
        }
        .animation(.linear(duration: 0.04), value: levels)
    }
}

/// The 192 embedding values as bars about a centre line: up for positive, down for negative.
struct VoiceprintView: View {
    var vector: [Float]
    var tint: Color

    var body: some View {
        Canvas { ctx, size in
            guard !vector.isEmpty else { return }
            let peak = CGFloat(vector.map { abs($0) }.max() ?? 1)
            let slot = size.width / CGFloat(vector.count)
            let w = max(1, slot * 0.7)
            let mid = size.height / 2
            for (i, v) in vector.enumerated() {
                let h = max(1, CGFloat(abs(v)) / peak * (mid - 1))
                let up = v >= 0
                let r = CGRect(x: CGFloat(i) * slot, y: up ? mid - h : mid, width: w, height: h)
                ctx.fill(Path(r), with: .color(up ? tint : tint.opacity(0.45)))
            }
            ctx.fill(Path(CGRect(x: 0, y: mid - 0.25, width: size.width, height: 0.5)), with: .color(.white.opacity(0.25)))
        }
        .accessibilityLabel("Voiceprint of \(vector.count) values")
    }
}

/// A 270° arc gauge for the verification score, with the decision threshold marked.
struct ScoreGauge: View {
    var score: Double
    var threshold: Double
    var color: Color

    private let sweep = 0.75

    var body: some View {
        ZStack {
            Circle()
                .trim(from: 0, to: sweep)
                .stroke(.white.opacity(0.10), style: StrokeStyle(lineWidth: 16, lineCap: .round))
                .rotationEffect(.degrees(135))
            Circle()
                .trim(from: 0, to: sweep * min(max(score, 0), 1))
                .stroke(AngularGradient(colors: [Theme.violet, Theme.cyan, color], center: .center,
                                        startAngle: .degrees(0), endAngle: .degrees(270)),
                        style: StrokeStyle(lineWidth: 16, lineCap: .round))
                .rotationEffect(.degrees(135))
                .shadow(color: color.opacity(0.5), radius: 10)
            // threshold tick
            GeometryReader { g in
                let r = min(g.size.width, g.size.height) / 2
                Capsule()
                    .fill(.white)
                    .frame(width: 3, height: 26)
                    .position(x: g.size.width / 2, y: g.size.height / 2 - r)
                    .rotationEffect(.degrees(-135 + 270 * threshold))
            }
            VStack(spacing: 2) {
                Text(score, format: .number.precision(.fractionLength(3)))
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .monospacedDigit()
                    .contentTransition(.numericText(value: score))
                    .foregroundStyle(.white)
                Text("score")
                    .font(.caption.weight(.semibold))
                    .textCase(.uppercase)
                    .kerning(1.2)
                    .foregroundStyle(.white.opacity(0.55))
            }
        }
        .animation(.spring(duration: 0.8, bounce: 0.25), value: score)
    }
}

struct StatChip: View {
    var symbol: String
    var text: String

    var body: some View {
        Label(text, systemImage: symbol)
            .font(.caption2.weight(.medium))
            .monospacedDigit()
            .foregroundStyle(.white.opacity(0.75))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.white.opacity(0.08), in: Capsule())
    }
}

/// The engine's status, for a toolbar.
struct EngineBadge: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 6) {
            switch model.status {
            case .downloading:
                Image(systemName: "arrow.down.circle.fill").foregroundStyle(Theme.cyan)
                DownloadPercent(downloader: model.downloader)
            case .compiling:
                ProgressView().controlSize(.mini).tint(.white)
                Text("Compiling model")
            case .loading:
                ProgressView().controlSize(.mini).tint(.white)
                Text("Preparing model")
            case .ready(let info):
                Circle().fill(Theme.match).frame(width: 7, height: 7)
                    .shadow(color: Theme.match, radius: 4)
                Text("\(info.runtime.label) · \(info.compute.short)")
            case .failed:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Theme.mismatch)
                Text("Model failed")
            }
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.white.opacity(0.85))
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }
}

private struct DownloadPercent: View {
    @ObservedObject var downloader: ModelDownloader

    var body: some View {
        Text("Downloading \(Int((downloader.fraction * 100).rounded())) %")
            .monospacedDigit()
            .contentTransition(.numericText())
    }
}

/// Under each screen's title while the selected runtime is not ready: the download (with its progress), the
/// one-time compile, the load, or what failed and a retry.
struct ModelStatusBanner: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Group {
            switch model.status {
            case .ready:
                EmptyView()
            case .downloading(let runtime):
                DownloadCard(runtime: runtime, downloader: model.downloader)
            case .compiling(let runtime):
                StepCard(symbol: "gearshape.2.fill", title: "Compiling \(runtime.label) for this device",
                         detail: "Once per download; the compiled model is kept.")
            case .loading:
                StepCard(symbol: "cpu", title: "Loading \(model.runtime.label)",
                         detail: model.runtime == .coreAI ? "The first load specializes the bundle for this device."
                                                          : "Preparing the model for the selected compute unit.")
            case .failed(let message):
                VStack(alignment: .leading, spacing: 10) {
                    Label("The model could not be loaded", systemImage: "exclamationmark.triangle.fill")
                        .font(.headline)
                        .foregroundStyle(Theme.mismatch)
                    Text(message)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.white.opacity(0.7))
                        .lineLimit(4)
                    Button("Try again", systemImage: "arrow.clockwise") { Task { await model.load() } }
                        .buttonStyle(.glass)
                }
                .card(tint: Theme.mismatch)
            }
        }
        .animation(.snappy, value: model.status)
    }
}

private struct DownloadCard: View {
    let runtime: Runtime
    @ObservedObject var downloader: ModelDownloader

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Downloading \(runtime.label)", systemImage: "arrow.down.circle.fill")
                    .font(.headline)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .foregroundStyle(.white)
                Spacer()
                Text("\(Int((downloader.fraction * 100).rounded())) %")
                    .font(.headline)
                    .monospacedDigit()
                    .foregroundStyle(Theme.cyan)
                    .contentTransition(.numericText())
            }
            ProgressView(value: downloader.fraction)
                .tint(Theme.cyan)
                .animation(.linear(duration: 0.2), value: downloader.fraction)
            HStack {
                Text(downloader.detail.isEmpty ? "Contacting Hugging Face…" : downloader.detail)
                    .monospacedDigit()
                Spacer()
                Text("from Hugging Face")
            }
            .font(.caption)
            .foregroundStyle(.white.opacity(0.55))
        }
        .card(tint: Theme.cyan)
    }
}

private struct StepCard: View {
    var symbol: String
    var title: String
    var detail: String

    var body: some View {
        HStack(spacing: 12) {
            ProgressView().tint(.white)
            VStack(alignment: .leading, spacing: 2) {
                Label(title, systemImage: symbol).font(.headline).foregroundStyle(.white)
                Text(detail).font(.caption).foregroundStyle(.white.opacity(0.55))
            }
        }
        .card()
    }
}

extension Double {
    /// 4.2 s, 1:03.5
    var clock: String {
        let m = Int(self) / 60
        let s = self - Double(m * 60)
        return m > 0 ? String(format: "%d:%04.1f", m, s) : String(format: "%.1f s", s)
    }
}
