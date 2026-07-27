// Copyright 2026 MacPaw Way Ltd.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
// EntityVisualization.swift
// displaCy-style inline highlighting of recognised entities, plus the flow layout and
// colour palette it needs.

import SwiftUI

// MARK: - Model

struct HighlightedEntity: Identifiable, Hashable {
    let id = UUID()
    let text: String
    let label: String
    let start: Int          // character (unicode-scalar) offsets, as the decoder reports
    let end: Int
    let confidence: Float?
}

// MARK: - Colour palette (stable per label)

enum EntityPalette {
    private static let colors: [Color] = [
        .blue, .green, .orange, .purple, .pink, .teal, .indigo, .red, .brown, .cyan,
    ]

    /// Deterministic colour for a label, so the same label is always the same colour.
    static func color(for label: String) -> Color {
        let hash = label.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0x7fff_ffff }
        return colors[hash % colors.count]
    }
}

// MARK: - Highlighted text

/// Renders `text` with each entity span shown as a coloured pill carrying its label —
/// the displaCy look. Non-entity text flows as plain words so the whole thing wraps.
struct EntityHighlightView: View {
    let text: String
    let entities: [HighlightedEntity]

    var body: some View {
        FlowLayout(spacing: 4, lineSpacing: 8) {
            ForEach(Array(segments.enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .word(let word):
                    Text(word).font(.body)
                case .entity(let entity):
                    EntityPill(entity: entity)
                }
            }
        }
    }

    private enum Segment {
        case word(String)
        case entity(HighlightedEntity)
    }

    /// Split the text into plain words and atomic entity pills, in reading order.
    private var segments: [Segment] {
        let scalars = Array(text.unicodeScalars)
        let sorted = entities.sorted { $0.start < $1.start }
        var result: [Segment] = []
        var cursor = 0

        func appendWords(_ range: Range<Int>) {
            guard range.lowerBound < range.upperBound,
                  range.upperBound <= scalars.count else { return }
            let piece = String(String.UnicodeScalarView(scalars[range]))
            for word in piece.split(whereSeparator: { $0 == " " || $0 == "\n" }) {
                result.append(.word(String(word)))
            }
        }

        for entity in sorted {
            guard entity.start >= cursor, entity.end <= scalars.count, entity.start < entity.end
            else { continue }
            appendWords(cursor ..< entity.start)
            result.append(.entity(entity))
            cursor = entity.end
        }
        appendWords(cursor ..< scalars.count)
        return result
    }
}

private struct EntityPill: View {
    let entity: HighlightedEntity

    var body: some View {
        let color = EntityPalette.color(for: entity.label)
        HStack(spacing: 4) {
            Text(entity.text).font(.body)
            Text(entity.label.uppercased())
                .font(.system(size: 9, weight: .heavy))
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(color.opacity(0.9), in: Capsule())
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 6).padding(.vertical, 2)
        .background(color.opacity(0.18), in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(color.opacity(0.5), lineWidth: 1))
    }
}

// MARK: - Legend

struct EntityLegend: View {
    let labels: [String]

    var body: some View {
        FlowLayout(spacing: 8, lineSpacing: 6) {
            ForEach(labels, id: \.self) { label in
                HStack(spacing: 4) {
                    Circle().fill(EntityPalette.color(for: label)).frame(width: 9, height: 9)
                    Text(label).font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - Flow layout (wraps children like text, iOS 16+ Layout)

struct FlowLayout: Layout {
    var spacing: CGFloat = 6
    var lineSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var rows = layout(subviews, maxWidth: maxWidth)
        let height = rows.last.map { $0.y + $0.height } ?? 0
        rows.removeAll()
        return CGSize(width: maxWidth == .infinity ? 0 : maxWidth, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) {
        let rows = layout(subviews, maxWidth: bounds.width)
        for row in rows {
            for item in row.items {
                subviews[item.index].place(
                    at: CGPoint(x: bounds.minX + item.x, y: bounds.minY + row.y),
                    proposal: ProposedViewSize(item.size))
            }
        }
    }

    private struct Row { var y: CGFloat; var height: CGFloat; var items: [Item] }
    private struct Item { var index: Int; var x: CGFloat; var size: CGSize }

    private func layout(_ subviews: Subviews, maxWidth: CGFloat) -> [Row] {
        var rows: [Row] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var items: [Item] = []

        for (index, subview) in subviews.enumerated() {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, !items.isEmpty {
                rows.append(Row(y: y, height: rowHeight, items: items))
                y += rowHeight + lineSpacing
                x = 0; rowHeight = 0; items = []
            }
            items.append(Item(index: index, x: x, size: size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        if !items.isEmpty { rows.append(Row(y: y, height: rowHeight, items: items)) }
        return rows
    }
}
