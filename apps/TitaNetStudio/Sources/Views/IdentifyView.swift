// IdentifyView — speaker identification: enroll named voices, then rank them against a new recording. Unfolded, the
// enrolled voices sit left of the fold and the question and its answer right of it.

import SwiftUI

struct IdentifyView: View {
    @Environment(AppModel.self) private var model
    @State private var enrolling = false

    var body: some View {
        AdaptiveSplit {
            VStack(alignment: .leading, spacing: 12) {
                ScreenHeader(title: "Identify", subtitle: "Enroll voices, then ask who is speaking.")
                ModelStatusBanner()
            }
        } leading: {
            enrolled
        } trailing: {
            ClipCard(slot: model.probe, title: "Who's speaking?", badge: "?", tint: Theme.violet)
            ranking
        }
        .sheet(isPresented: $enrolling) { EnrollSheet() }
    }

    private var enrolled: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Enrolled voices").font(.headline).foregroundStyle(.white)
                Text("\(model.speakers.count)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white.opacity(0.6))
                    .padding(.horizontal, 7).padding(.vertical, 2)
                    .background(.white.opacity(0.1), in: Capsule())
                Spacer()
                Button("Enroll", systemImage: "plus") { enrolling = true }
                    .buttonStyle(.glassProminent)
                    .tint(Theme.violet.opacity(0.85))
                    .fontWeight(.semibold)
            }
            if model.speakers.isEmpty {
                Text("No one enrolled yet. Enroll two or three people with a few seconds each, then ask who's speaking.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.55))
            } else {
                ForEach(model.speakers) { s in
                    SpeakerRow(speaker: s)
                        .contextMenu {
                            Button("Remove \(s.name)", systemImage: "trash", role: .destructive) {
                                withAnimation { model.delete(s) }
                            }
                        }
                }
            }
        }
        .card()
    }

    @ViewBuilder private var ranking: some View {
        let r = model.ranking
        if !r.isEmpty, let top = r.first {
            let known = top.score >= model.threshold
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 12) {
                    Avatar(speaker: known ? top.speaker : nil, size: 46)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(known ? top.speaker.name : "Unknown speaker")
                            .font(.system(.title2, design: .rounded, weight: .bold))
                            .foregroundStyle(.white)
                        Text(known ? String(format: "Best match · score %.3f", top.score)
                                   : String(format: "No one above %.2f", model.threshold))
                            .font(.subheadline)
                            .foregroundStyle(known ? Theme.match : Theme.mismatch)
                    }
                }
                ForEach(r, id: \.speaker.id) { item in
                    ScoreBar(speaker: item.speaker, score: item.score, threshold: model.threshold)
                }
            }
            .card(tint: known ? Theme.match : Theme.mismatch)
            .sensoryFeedback(known ? .success : .warning, trigger: top.score)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }
}

struct Avatar: View {
    var speaker: Speaker?
    var size: CGFloat

    var body: some View {
        ZStack {
            Circle().fill((speaker?.color ?? .gray).gradient)
            if let speaker {
                Text(speaker.initials)
                    .font(.system(size: size * 0.38, weight: .bold, design: .rounded))
                    .foregroundStyle(.black.opacity(0.75))
            } else {
                Image(systemName: "questionmark")
                    .font(.system(size: size * 0.42, weight: .bold))
                    .foregroundStyle(.black.opacity(0.6))
            }
        }
        .frame(width: size, height: size)
    }
}

struct SpeakerRow: View {
    var speaker: Speaker

    var body: some View {
        HStack(spacing: 12) {
            Avatar(speaker: speaker, size: 38)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(speaker.name).font(.subheadline.weight(.semibold)).foregroundStyle(.white)
                    Spacer()
                    Text(speaker.seconds.clock).font(.caption).monospacedDigit().foregroundStyle(.white.opacity(0.5))
                }
                VoiceprintView(vector: speaker.embedding, tint: speaker.color)
                    .frame(height: 18)
            }
        }
        .padding(10)
        .background(.white.opacity(0.05), in: .rect(cornerRadius: 16))
    }
}

struct ScoreBar: View {
    var speaker: Speaker
    var score: Double
    var threshold: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(speaker.name).font(.caption.weight(.semibold)).foregroundStyle(.white.opacity(0.85))
                Spacer()
                Text(score, format: .number.precision(.fractionLength(3)))
                    .font(.caption.weight(.semibold)).monospacedDigit()
                    .foregroundStyle(score >= threshold ? Theme.match : .white.opacity(0.55))
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.white.opacity(0.08))
                    Capsule().fill(speaker.color.gradient)
                        .frame(width: g.size.width * min(max(score, 0), 1))
                    Rectangle().fill(.white.opacity(0.8))
                        .frame(width: 2)
                        .offset(x: g.size.width * threshold - 1)
                }
            }
            .frame(height: 8)
        }
        .animation(.spring(duration: 0.7), value: score)
    }
}

struct EnrollSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var slot = ClipSlot()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    TextField("Name", text: $name)
                        .textInputAutocapitalization(.words)
                        .font(.title3.weight(.semibold))
                        .padding(16)
                        .glassEffect(.regular, in: .rect(cornerRadius: 20))
                    ClipCard(slot: slot, title: "Voice sample", badge: initials, tint: Theme.violet)
                    Text("Read a sentence or two in a normal voice. Longer, cleaner samples identify more reliably.")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.5))
                }
                .padding(16)
            }
            .background { Backdrop() }
            .navigationTitle("Enroll a voice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", systemImage: "xmark") { model.clear(slot); dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save", systemImage: "checkmark") {
                        if let clip = slot.clip { model.enroll(name: trimmed, clip: clip) }
                        dismiss()
                    }
                    .disabled(trimmed.isEmpty || slot.clip == nil)
                }
            }
        }
        .presentationBackground(.clear)
    }

    private var trimmed: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var initials: String { Speaker(name: trimmed, hue: 0, embedding: [], seconds: 0).initials }
}
