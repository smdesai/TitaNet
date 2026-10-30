// TitaNet Studio — try the titanet-lean Core AI bundle on a phone: compare two voices, or identify a speaker against
// enrolled ones, from the microphone or an audio file.

import SwiftUI

@main
struct StudioApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .preferredColorScheme(.dark)
                .task {
                    await model.load()
                    if UserDefaults.standard.bool(forKey: "selftest") { await model.selfTest() }
                }
        }
    }
}

struct RootView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.horizontalSizeClass) private var sizeClass
    @State private var window: CGRect = .zero
    @State private var fold: CGPoint?
    @State private var hinge = HingeState()
    // `-tab identify|model` at launch picks the first tab (device screenshots without touching the phone)
    @State private var tab = UserDefaults.standard.string(forKey: "tab") ?? "compare"

    private var posture: Posture { .resolve(window: window.size, sizeClass: sizeClass, hinge: hinge) }

    var body: some View {
        @Bindable var model = model
        TabView(selection: $tab) {
            Tab("Compare", systemImage: "person.2.wave.2", value: "compare") { screen { CompareView() } }
            Tab("Identify", systemImage: "person.crop.circle.badge.questionmark", value: "identify") { screen { IdentifyView() } }
            Tab("Model", systemImage: "cpu", value: "model") { screen { SettingsView() } }
        }
        .tint(Theme.cyan)
        .environment(\.posture, posture)
        .environment(\.window, window)
        .environment(\.fold, fold)
        .environment(\.hinge, hinge)
        .background {
            Color.clear
                .ignoresSafeArea()
                .onGeometryChange(for: CGRect.self) { $0.frame(in: .global) } action: { window = $0 }
            FoldReader { fold = $0 }.ignoresSafeArea()
        }
        .trackHinge($hinge)
        .onChange(of: "\(posture) \(Int(window.width))x\(Int(window.height)) \(sizeClass == .regular ? "regular" : "compact") \(hinge) fold \(fold.map { "\(Int($0.x)),\(Int($0.y))" } ?? "-")", initial: true) { _, s in
            print("[studio] layout: \(s)")
        }
        .alert("Something went wrong", isPresented: Binding(get: { model.alert != nil }, set: { if !$0 { model.alert = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.alert ?? "")
        }
    }

    private func screen<V: View>(@ViewBuilder _ v: () -> V) -> some View {
        NavigationStack {
            // a ZStack, so the backdrop fills under the bars whatever the content's own extent
            ZStack {
                Backdrop()
                v()
            }
            .toolbar { ToolbarItem(placement: .topBarTrailing) { EngineBadge() } }
        }
    }
}
