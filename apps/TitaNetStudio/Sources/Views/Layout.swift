// Layout — folded vs unfolded (iPhone Duo and other foldables).
//
//   folded    the narrow display: one scrolling column, as on any iPhone
//   unfolded  the wide display, flat: two panes that meet at the fold, each scrolling on its own
//   book      the wide display, partly open: the same two panes with a wide gutter over the crease; with the hinge
//             across the screen (taller than wide), the panes stack above and below it instead
//
// The posture comes from the window's horizontal size class and width, refined by SwiftUI's hinge state where the OS
// has it (onHingeChange, iOS 27.1). The fold runs through the middle of the physical screen, which is not the middle
// of the app's window (on iPhone Duo the status column sits outside it), so FoldReader converts the screen's centre
// into window coordinates and the panes are cut there.

import SwiftUI
import UIKit

struct HingeState: Equatable, CustomStringConvertible {
    enum Status: String { case none, closed, partial, open }
    var status: Status = .none
    var degrees: Double = 0

    var description: String { status == .none ? "no hinge" : "\(status.rawValue) \(Int(degrees.rounded()))°" }
}

enum Posture: String, Equatable {
    case folded, unfolded, book

    static func resolve(window: CGSize, sizeClass: UserInterfaceSizeClass?, hinge: HingeState) -> Posture {
        if hinge.status == .closed { return .folded }
        guard sizeClass == .regular || window.width >= 600 else { return .folded }
        return hinge.status == .partial ? .book : .unfolded
    }

    var label: String {
        switch self {
        case .folded: "Folded"
        case .unfolded: "Unfolded"
        case .book: "Book (partly open)"
        }
    }
}

extension EnvironmentValues {
    @Entry var posture: Posture = .folded
    @Entry var window: CGRect = .zero
    @Entry var fold: CGPoint? = nil          // the screen's centre, in window (SwiftUI global) coordinates
    @Entry var hinge = HingeState()
}

/// Reports the physical screen's centre in its window's coordinates, again on every layout (rotation, fold, resize).
struct FoldReader: UIViewRepresentable {
    var onChange: (CGPoint) -> Void

    final class Probe: UIView {
        var onChange: ((CGPoint) -> Void)?
        private var last: CGPoint?

        override func layoutSubviews() {
            super.layoutSubviews()
            report()
        }

        override func didMoveToWindow() {
            super.didMoveToWindow()
            report()
        }

        private func report() {
            guard let window, let screen = window.windowScene?.screen else { return }
            let b = screen.bounds
            let p = window.convert(CGPoint(x: b.midX, y: b.midY), from: screen.coordinateSpace)
            guard p != last else { return }
            last = p
            let cb = onChange
            DispatchQueue.main.async { cb?(p) }
        }
    }

    func makeUIView(context: Context) -> Probe {
        let v = Probe()
        v.isUserInteractionEnabled = false
        v.onChange = onChange
        return v
    }

    func updateUIView(_ v: Probe, context: Context) { v.onChange = onChange }
}

extension View {
    /// Keeps `state` in step with the device hinge, where the OS reports one.
    @ViewBuilder func trackHinge(_ state: Binding<HingeState>) -> some View {
        if #available(iOS 27.1, *) {
            onHingeChange { _, new in
                guard let h = new.hinge else { state.wrappedValue = HingeState(); return }
                let status: HingeState.Status = switch h.status {
                case .closed: .closed
                case .partiallyOpen: .partial
                case .fullyOpen: .open
                default: .none
                }
                state.wrappedValue = HingeState(status: status, degrees: h.angle.degrees)
            }
        } else {
            self
        }
    }
}

/// A screen in either posture: `header`, then `leading` and `trailing` in one column (folded) or in two panes split at
/// the fold (unfolded, book).
struct AdaptiveSplit<Header: View, Leading: View, Trailing: View>: View {
    @Environment(\.posture) private var posture
    @Environment(\.window) private var window
    @Environment(\.fold) private var foldPoint
    private let header: Header
    private let leading: Leading
    private let trailing: Trailing

    init(@ViewBuilder header: () -> Header, @ViewBuilder leading: () -> Leading, @ViewBuilder trailing: () -> Trailing) {
        self.header = header()
        self.leading = leading()
        self.trailing = trailing()
    }

    var body: some View {
        GeometryReader { g in
            let frame = g.frame(in: .global)
            let center = foldPoint ?? CGPoint(x: window.midX, y: window.midY)
            let gutter: CGFloat = posture == .book ? 48 : 24
            Group {
                switch posture {
                case .folded:
                    pane(spacing: 16) { header; leading; trailing }
                        .padding(.horizontal, 16)
                case .unfolded, .book:
                    if posture == .book && window.height > window.width {
                        // hinge across the screen: top and bottom halves
                        let fold = clamp(center.y - frame.minY, g.size.height)
                        VStack(spacing: gutter) {
                            pane(spacing: 16) { header; leading }.frame(height: max(0, fold - gutter / 2))
                            pane(spacing: 16) { trailing }
                        }
                        .padding(.horizontal, 24)
                    } else {
                        let fold = clamp(center.x - frame.minX, g.size.width)
                        let pad: CGFloat = 24
                        VStack(alignment: .leading, spacing: 8) {
                            header.padding(.horizontal, pad)
                            HStack(alignment: .top, spacing: gutter) {
                                pane(spacing: 16) { leading }
                                    .frame(width: max(0, fold - pad - gutter / 2))
                                pane(spacing: 16) { trailing }
                                    .frame(width: max(0, g.size.width - fold - pad - gutter / 2))
                            }
                            .padding(.horizontal, pad)
                        }
                    }
                }
            }
            .animation(.smooth(duration: 0.45), value: posture)
        }
    }

    /// The fold's offset within this view, kept off the edges if the window is not where it is expected.
    private func clamp(_ v: CGFloat, _ extent: CGFloat) -> CGFloat { min(max(v, extent * 0.35), extent * 0.65) }

    private func pane<V: View>(spacing: CGFloat, @ViewBuilder _ content: () -> V) -> some View {
        ScrollView {
            VStack(spacing: spacing) { content() }
                .padding(.bottom, 32)
        }
        .scrollIndicators(.hidden)
    }
}
