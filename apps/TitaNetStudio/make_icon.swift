// Renders TitaNet Studio's app icon: the score gauge's 270° arc around a voiceprint waveform, in the app's palette.
//   swift apps/TitaNetStudio/make_icon.swift <out.png>
// Writes a 1024 x 1024 opaque PNG (the single size an iOS asset catalog needs).

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let S: CGFloat = 1024
let cs = CGColorSpace(name: CGColorSpace.displayP3)!
let ctx = CGContext(data: nil, width: Int(S), height: Int(S), bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!

func rgb(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor { CGColor(colorSpace: cs, components: [r, g, b, a])! }
let cyan = rgb(0.33, 0.86, 1.0), violet = rgb(0.62, 0.47, 1.0), pink = rgb(1.0, 0.42, 0.72)

// background: deep indigo, lit from the upper left, with a violet glow behind the mark
let bg = CGGradient(colorsSpace: cs, colors: [rgb(0.13, 0.10, 0.34), rgb(0.05, 0.05, 0.16), rgb(0.02, 0.02, 0.07)] as CFArray,
                    locations: [0, 0.55, 1])!
ctx.drawLinearGradient(bg, start: CGPoint(x: 0, y: S), end: CGPoint(x: S, y: 0), options: [])
let glow = CGGradient(colorsSpace: cs, colors: [rgb(0.45, 0.30, 1.0, 0.55), rgb(0.45, 0.30, 1.0, 0)] as CFArray, locations: [0, 1])!
let c = CGPoint(x: S / 2, y: S / 2)
ctx.drawRadialGradient(glow, startCenter: c, startRadius: 0, endCenter: c, endRadius: S * 0.48, options: [])

/// Fill the current clip with the cyan -> violet -> pink sweep, left to right.
func sweep(_ path: CGPath) {
    ctx.saveGState()
    ctx.addPath(path)
    ctx.clip()
    let g = CGGradient(colorsSpace: cs, colors: [cyan, violet, pink] as CFArray, locations: [0, 0.55, 1])!
    ctx.drawLinearGradient(g, start: CGPoint(x: S * 0.18, y: S * 0.5), end: CGPoint(x: S * 0.82, y: S * 0.5),
                           options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
    ctx.restoreGState()
}

// the gauge: a 270° track open at the bottom, as ScoreGauge draws it, filled clockwise from the lower left to a score
// of 0.85 (past the 0.70 threshold: same speaker)
let R = S * 0.335, lw = S * 0.062
let start = CGFloat.pi * 5 / 4, score: CGFloat = 0.85
let track = CGMutablePath()
track.addArc(center: c, radius: R, startAngle: start, endAngle: -.pi / 4, clockwise: true)   // y-up: through the top
ctx.addPath(track.copy(strokingWithWidth: lw, lineCap: .round, lineJoin: .round, miterLimit: 1))
ctx.setFillColor(rgb(1, 1, 1, 0.10))
ctx.fillPath()
let arc = CGMutablePath()
arc.addArc(center: c, radius: R, startAngle: start, endAngle: start - score * .pi * 3 / 2, clockwise: true)
let arcStroke = arc.copy(strokingWithWidth: lw, lineCap: .round, lineJoin: .round, miterLimit: 1)
ctx.setShadow(offset: .zero, blur: 40, color: rgb(0.55, 0.45, 1.0, 0.9))
sweep(arcStroke)
ctx.setShadow(offset: .zero, blur: 0, color: nil)

// the threshold tick at 0.70 of the sweep, in white
let tickA = CGFloat.pi * 5 / 4 - 0.70 * (CGFloat.pi * 3 / 2)
let tick = CGMutablePath()
tick.move(to: CGPoint(x: c.x + (R - lw * 0.95) * cos(tickA), y: c.y + (R - lw * 0.95) * sin(tickA)))
tick.addLine(to: CGPoint(x: c.x + (R + lw * 0.95) * cos(tickA), y: c.y + (R + lw * 0.95) * sin(tickA)))
ctx.addPath(tick.copy(strokingWithWidth: S * 0.018, lineCap: .round, lineJoin: .round, miterLimit: 1))
ctx.setFillColor(rgb(1, 1, 1, 0.95))
ctx.fillPath()

// the voiceprint: mirrored rounded bars under a speech-like envelope
let heights: [CGFloat] = [0.16, 0.30, 0.52, 0.36, 0.74, 1.0, 0.62, 0.86, 0.44, 0.58, 0.28, 0.14]
let barW = S * 0.030, gap = S * 0.020
let total = CGFloat(heights.count) * barW + CGFloat(heights.count - 1) * gap
let bars = CGMutablePath()
for (i, h) in heights.enumerated() {
    let x = c.x - total / 2 + CGFloat(i) * (barW + gap)
    let hh = max(barW, h * S * 0.30)
    bars.addRoundedRect(in: CGRect(x: x, y: c.y - hh / 2 + S * 0.01, width: barW, height: hh),
                        cornerWidth: barW / 2, cornerHeight: barW / 2)
}
ctx.setShadow(offset: .zero, blur: 28, color: rgb(0.4, 0.8, 1.0, 0.7))
sweep(bars)

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
precondition(CGImageDestinationFinalize(dest))
print("wrote \(url.path)")
