// Theme — colors, the animated backdrop, and the glass card every section sits on.

import SwiftUI

enum Theme {
    static let cyan = Color(red: 0.33, green: 0.86, blue: 1.0)
    static let violet = Color(red: 0.62, green: 0.47, blue: 1.0)
    static let pink = Color(red: 1.0, green: 0.42, blue: 0.72)
    static let match = Color(red: 0.32, green: 0.92, blue: 0.62)
    static let mismatch = Color(red: 1.0, green: 0.62, blue: 0.30)
    static let record = Color(red: 1.0, green: 0.30, blue: 0.36)

    static let accent = LinearGradient(colors: [cyan, violet], startPoint: .leading, endPoint: .trailing)
}

struct Backdrop: View {
    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 20)) { tl in
            let t = tl.date.timeIntervalSinceReferenceDate
            let s = Float(sin(t * 0.23)) * 0.08, c = Float(cos(t * 0.19)) * 0.08
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [0.5 + s, 0], [1, 0],
                    [0, 0.5 + c], [0.5 - c, 0.5 + s], [1, 0.5 - s],
                    [0, 1], [0.5 + c, 1], [1, 1],
                ],
                colors: [
                    Color(red: 0.05, green: 0.06, blue: 0.16), Color(red: 0.10, green: 0.07, blue: 0.26),
                    Color(red: 0.04, green: 0.10, blue: 0.22),
                    Color(red: 0.06, green: 0.12, blue: 0.28), Color(red: 0.16, green: 0.08, blue: 0.32),
                    Color(red: 0.03, green: 0.05, blue: 0.14),
                    Color(red: 0.02, green: 0.03, blue: 0.08), Color(red: 0.07, green: 0.04, blue: 0.15),
                    Color(red: 0.02, green: 0.06, blue: 0.12),
                ]
            )
        }
        .ignoresSafeArea()
    }
}

struct Card: ViewModifier {
    var tint: Color?

    func body(content: Content) -> some View {
        content
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .glassEffect(.regular.tint(tint?.opacity(0.10)), in: .rect(cornerRadius: 28))
    }
}

extension View {
    func card(tint: Color? = nil) -> some View { modifier(Card(tint: tint)) }
}

/// A screen title with a subtitle, left-aligned above the cards.
struct ScreenHeader: View {
    var title: String
    var subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(.largeTitle, design: .rounded, weight: .bold))
                .foregroundStyle(.white)
            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.6))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 8)
    }
}
