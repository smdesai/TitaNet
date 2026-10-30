// CompareView — speaker verification: two clips, one score, one verdict. Unfolded, the clips sit left of the fold and
// the verdict right of it.

import SwiftUI

struct CompareView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.posture) private var posture

    var body: some View {
        AdaptiveSplit {
            VStack(alignment: .leading, spacing: 12) {
                ScreenHeader(title: "Compare", subtitle: "Are these two recordings the same speaker?")
                ModelStatusBanner()
            }
        } leading: {
            ClipCard(slot: model.a, title: "Voice A", badge: "A", tint: Theme.cyan)
            ClipCard(slot: model.b, title: "Voice B", badge: "B", tint: Theme.pink)
        } trailing: {
            result
        }
    }

    @ViewBuilder private var result: some View {
        if let score = model.score(model.a, model.b) {
            ResultCard(score: score, threshold: model.threshold, large: posture != .folded)
                .transition(.move(edge: .bottom).combined(with: .opacity))
        } else if posture == .folded {
            HStack(spacing: 12) {
                Image(systemName: "person.wave.2")
                    .font(.title2)
                    .foregroundStyle(Theme.accent)
                Text("Add a recording or file to both slots and the verdict appears here.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.6))
            }
            .card()
        } else {
            VerdictPlaceholder(ready: [model.a.clip != nil, model.b.clip != nil])
        }
    }
}

/// The unfolded verdict pane before both clips are in: an empty gauge and what is still missing.
struct VerdictPlaceholder: View {
    var ready: [Bool]

    var body: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle()
                    .trim(from: 0, to: 0.75)
                    .stroke(.white.opacity(0.10), style: StrokeStyle(lineWidth: 18, lineCap: .round))
                    .rotationEffect(.degrees(135))
                Image(systemName: "person.wave.2")
                    .font(.system(size: 52, weight: .light))
                    .foregroundStyle(Theme.accent)
                    .symbolEffect(.pulse, options: .repeat(.continuous))
            }
            .frame(width: 240, height: 240)
            Text("Verdict").font(.system(.title2, design: .rounded, weight: .bold)).foregroundStyle(.white)
            VStack(alignment: .leading, spacing: 8) {
                step("Voice A", done: ready[0], tint: Theme.cyan)
                step("Voice B", done: ready[1], tint: Theme.pink)
            }
            Text("Record or import a clip on the left for each voice.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.55))
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .card()
    }

    private func step(_ title: String, done: Bool, tint: Color) -> some View {
        Label(title, systemImage: done ? "checkmark.circle.fill" : "circle.dashed")
            .font(.headline)
            .foregroundStyle(done ? tint : .white.opacity(0.5))
            .contentTransition(.symbolEffect(.replace))
    }
}

struct ResultCard: View {
    var score: Double
    var threshold: Double
    var large = false

    private var same: Bool { score >= threshold }
    private var color: Color { same ? Theme.match : Theme.mismatch }

    var body: some View {
        VStack(spacing: large ? 20 : 14) {
            ScoreGauge(score: score, threshold: threshold, color: color)
                .frame(width: large ? 260 : 200, height: large ? 260 : 200)
                .padding(.top, 6)
            Label(same ? "Same speaker" : "Different speakers",
                  systemImage: same ? "checkmark.seal.fill" : "xmark.seal.fill")
                .font(.system(large ? .title : .title2, design: .rounded, weight: .bold))
                .foregroundStyle(color)
                .contentTransition(.symbolEffect(.replace))
            HStack(spacing: 8) {
                StatChip(symbol: "angle", text: String(format: "cosine %.3f", 2 * score - 1))
                StatChip(symbol: "line.diagonal", text: String(format: "threshold %.2f", threshold))
            }
            Text("Score is NeMo's (cos + 1) / 2 over the two 192-d embeddings. The white tick marks the decision threshold.")
                .font(.caption)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.45))
        }
        .padding(.vertical, large ? 12 : 0)
        .frame(maxWidth: .infinity)
        .card(tint: color)
        .sensoryFeedback(same ? .success : .warning, trigger: score)
    }
}
