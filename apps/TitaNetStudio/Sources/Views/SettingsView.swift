// SettingsView — compute unit, decision threshold, what the loaded bundle is, and the display posture.

import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.posture) private var posture
    @Environment(\.window) private var window
    @Environment(\.hinge) private var hinge
    @Environment(\.fold) private var fold
    @Environment(\.horizontalSizeClass) private var sizeClass

    var body: some View {
        @Bindable var model = model
        AdaptiveSplit {
            VStack(alignment: .leading, spacing: 12) {
                ScreenHeader(title: "Model", subtitle: "TitaNet-Large on \(model.runtime.label), on this device.")
                ModelStatusBanner()
            }
        } leading: {
            VStack(alignment: .leading, spacing: 12) {
                Text("Runtime").font(.headline).foregroundStyle(.white)
                Picker("Runtime", selection: $model.runtime) {
                    ForEach(Runtime.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                .disabled(model.isFetching)
                Text(runtimeNote)
                    .font(.caption).foregroundStyle(.white.opacity(0.5))
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Runtime.allCases) { r in
                        Label {
                            Text("\(r.label): ") + Text(storage(r)).foregroundStyle(.white.opacity(0.5))
                        } icon: {
                            Image(systemName: ModelStore.isReady(r) ? "checkmark.circle.fill" : "icloud.and.arrow.down")
                                .foregroundStyle(ModelStore.isReady(r) ? Theme.match : .white.opacity(0.5))
                        }
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.8))
                    }
                }
            }
            .card()

            VStack(alignment: .leading, spacing: 12) {
                Text("Compute unit").font(.headline).foregroundStyle(.white)
                Picker("Compute unit", selection: $model.compute) {
                    ForEach(ComputeUnit.allCases) { Text($0.short).tag($0) }
                }
                .pickerStyle(.segmented)
                .disabled(model.isFetching)
                Text(computeNote)
                    .font(.caption).foregroundStyle(.white.opacity(0.5))
            }
            .card()

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Decision threshold").font(.headline).foregroundStyle(.white)
                    Spacer()
                    Text(model.threshold, format: .number.precision(.fractionLength(2)))
                        .font(.headline).monospacedDigit().foregroundStyle(Theme.cyan)
                }
                Slider(value: $model.threshold, in: 0.5...0.95, step: 0.01)
                    .tint(Theme.cyan)
                HStack {
                    Text("Lenient").font(.caption2)
                    Spacer()
                    Button("Reset to 0.70") { model.threshold = Scoring.threshold }
                        .font(.caption.weight(.semibold))
                        .disabled(model.threshold == Scoring.threshold)
                    Spacer()
                    Text("Strict").font(.caption2)
                }
                .foregroundStyle(.white.opacity(0.5))
            }
            .card()

            displayCard
        } trailing: {
            bundleCard
        }
    }

    @ViewBuilder private var bundleCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Bundle").font(.headline).foregroundStyle(.white)
            switch model.status {
            case .downloading, .compiling:
                Text("Details appear once the model is downloaded and loaded.")
                    .font(.subheadline).foregroundStyle(.white.opacity(0.7))
            case .loading:
                HStack(spacing: 10) {
                    ProgressView().tint(.white)
                    Text("Loading the model on this device.")
                        .font(.subheadline).foregroundStyle(.white.opacity(0.7))
                }
            case .failed(let message):
                Text(message).font(.footnote.monospaced()).foregroundStyle(Theme.mismatch)
                Button("Try again") { Task { await model.load() } }.buttonStyle(.glass)
            case .ready(let info):
                row("Runtime", info.runtime.label)
                row("Recipe", info.recipe)
                row("Window", String(format: "%d frames · %.2f s", info.window, info.windowSeconds))
                row("Margin", "\(info.margin) frames each side")
                row("Functions", info.entrypoints.joined(separator: ", "))
                row("Front end", "\(info.mels)-mel log filterbank, 16 kHz")
                row("Device", info.architecture)
                row("Load", String(format: "%.2f s on %@", info.loadSeconds, info.compute.label))
            }
        }
        .card()
    }

    private var displayCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Display").font(.headline).foregroundStyle(.white)
                Spacer()
                Label(posture.label, systemImage: posture == .folded ? "iphone" : "book.pages")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.cyan)
                    .contentTransition(.symbolEffect(.replace))
            }
            row("Window", "\(Int(window.width)) × \(Int(window.height)) pt")
            row("Width class", sizeClass == .regular ? "regular" : "compact")
            row("Hinge", hinge.description)
            row("Screen centre", fold.map { "\(Int($0.x)), \(Int($0.y)) pt in window" } ?? "—")
        }
        .card()
    }

    /// Whether a runtime's model is on this device. Re-read whenever the status changes (a download or compile ends).
    private func storage(_ r: Runtime) -> String {
        _ = model.status
        if ModelStore.isReady(r) { return "downloaded" }
        if case .downloading(let d) = model.status, d == r { return "downloading now" }
        return "downloads \(ModelStore.size(r)) from Hugging Face when selected"
    }

    /// What each runtime is, with the device-gate numbers behind it (iPhone19,4, 40 VoxCeleb clips vs NeMo).
    private var runtimeNote: String {
        switch model.runtime {
        case .coreAI: "The 47 MB Core AI bundle: fp16 weights, fp32 compute. Worst cosine to NeMo 0.9999987."
        case .coreMLFP32: "The 88 MB Core ML model, fp32. Worst cosine to NeMo 0.9999999994 on the GPU."
        case .coreMLFP16: "The 47 MB Core ML model (fp16c), built for the Neural Engine: about 5× faster, worst cosine to NeMo 0.9968."
        }
    }

    private var computeNote: String {
        switch (model.runtime, model.compute) {
        case (.coreMLFP32, .ane): "Core ML runs fp32 on the GPU or CPU only, so this behaves like GPU. Changing it reloads the model."
        case (.coreMLFP16, .ane): "All of the fp16 scan runs on the Neural Engine. Changing it reloads the model."
        default: "Changing it reloads the model. The accuracy gates ran on the GPU (and the Neural Engine for fp16)."
        }
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(k).font(.subheadline).foregroundStyle(.white.opacity(0.55))
            Spacer()
            Text(v).font(.subheadline.weight(.medium)).monospacedDigit().foregroundStyle(.white)
                .multilineTextAlignment(.trailing)
        }
    }
}
