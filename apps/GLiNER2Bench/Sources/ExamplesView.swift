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
// ExamplesView.swift
// Interactive entity-recognition demo: type (or pick) text, run the model, and see the
// predictions highlighted displaCy-style. Uses its own persistently-loaded model, separate
// from the benchmark's load-and-release model.

import SwiftUI
import GLiNER2Swift

@MainActor
final class ExamplesModel: ObservableObject {
    @Published var status = "Tap Run to load the model"
    @Published var isBusy = false
    @Published var entities: [HighlightedEntity] = []
    @Published var lastText = ""
    @Published var errorMessage: String?

    private var model: GLiNER2?

    func run(text: String, labels: [String]) {
        guard !isBusy else { return }
        isBusy = true
        errorMessage = nil
        Task {
            do {
                if model == nil {
                    status = "Loading model…"
                    let path = try await ModelLocator.resolve { _, message in
                        Task { @MainActor in self.status = message }
                    }
                    model = try await GLiNER2.fromPretrained(path)
                }
                status = "Extracting…"
                entities = extract(text: text, labels: labels)
                lastText = text
                status = "\(entities.count) entit\(entities.count == 1 ? "y" : "ies") found"
            } catch {
                errorMessage = error.localizedDescription
                status = "Failed"
            }
            isBusy = false
        }
    }

    private func extract(text: String, labels: [String]) -> [HighlightedEntity] {
        guard let model else { return [] }
        let schema = model.createSchema().entities(labels)
        let result = model.extract(text: text, schema: schema, threshold: 0.5,
                                   includeConfidence: true, includeSpans: true)
        guard let entities = result["entities"] as? [String: [Any]] else { return [] }

        var found: [HighlightedEntity] = []
        for (label, spans) in entities {
            for span in spans {
                guard let dict = span as? [String: Any],
                      let text = dict["text"] as? String,
                      let start = dict["start"] as? Int,
                      let end = dict["end"] as? Int else { continue }
                found.append(HighlightedEntity(
                    text: text, label: label, start: start, end: end,
                    confidence: dict["confidence"] as? Float))
            }
        }
        return found.sorted { $0.start < $1.start }
    }
}

struct ExamplesView: View {
    @StateObject private var model = ExamplesModel()

    @State private var text = presets[0].text
    @State private var labelsText = "person, organization, location, date, product, title"

    private static let presets: [(name: String, text: String)] = [
        ("Tech", "Tim Cook is the CEO of Apple in Cupertino, and Satya Nadella leads Microsoft in Redmond."),
        ("News", "Maria Gonzalez joined Siemens in Munich in March 2021 after a decade at Bosch."),
        ("Sport", "Lionel Messi scored twice for Inter Miami against Orlando City on Saturday."),
        ("Science", "Dr. Chen published the results in Nature while working at Stanford University."),
    ]

    private var labels: [String] {
        labelsText.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            .filter { !$0.isEmpty }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    presetRow

                    field("Text") {
                        TextEditor(text: $text)
                            .frame(minHeight: 90)
                            .padding(6)
                            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                    }

                    field("Entity types (comma-separated)") {
                        TextField("person, organization, …", text: $labelsText)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .padding(8)
                            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                    }

                    runButton
                    statusRow
                    if let error = model.errorMessage {
                        Text(error).font(.footnote).foregroundStyle(.red)
                    }

                    if !model.lastText.isEmpty {
                        resultCard
                    }
                }
                .padding()
            }
            .navigationTitle("Predictions")
        }
    }

    private var presetRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Self.presets, id: \.name) { preset in
                    Button(preset.name) { text = preset.text }
                        .font(.subheadline)
                        .buttonStyle(.bordered)
                        .disabled(model.isBusy)
                }
            }
        }
    }

    private var runButton: some View {
        Button {
            model.run(text: text, labels: labels)
        } label: {
            Text(model.isBusy ? "Working…" : "Run recognition")
                .frame(maxWidth: .infinity).padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .disabled(model.isBusy || labels.isEmpty || text.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    private var statusRow: some View {
        HStack(spacing: 10) {
            if model.isBusy { ProgressView() }
            Text(model.status).font(.subheadline).foregroundStyle(.secondary)
        }
    }

    private var resultCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            EntityHighlightView(text: model.lastText, entities: model.entities)

            if !model.entities.isEmpty {
                Divider()
                EntityLegend(labels: Array(Set(model.entities.map(\.label))).sorted())
                Divider()
                ForEach(model.entities) { entity in
                    HStack {
                        Text(entity.text).fontWeight(.medium)
                        Text(entity.label.uppercased())
                            .font(.system(size: 9, weight: .heavy))
                            .foregroundStyle(EntityPalette.color(for: entity.label))
                        Spacer()
                        if let c = entity.confidence {
                            Text(String(format: "%.0f%%", c * 100))
                                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                    .font(.footnote)
                }
            } else {
                Text("No entities above threshold.").font(.footnote).foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
    }

    private func field<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            content()
        }
    }
}
