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
// BenchmarkView.swift
// The whole UI: pick precision(s), run, watch progress, read the table, share the results.
// A deliberately simple "speedtest"-style screen.

import SwiftUI

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var status: String = "Ready"
    @Published var isRunning = false
    @Published var downloadProgress: Double?
    @Published var reports: [BenchmarkReport] = []
    @Published var errorMessage: String?

    private let engine = BenchmarkEngine()

    func run(precisions: [Precision]) {
        guard !isRunning else { return }
        isRunning = true
        errorMessage = nil
        reports = []

        Task {
            do {
                let modelPath = try await ModelLocator.resolve { [weak self] fraction, message in
                    Task { @MainActor in
                        self?.downloadProgress = fraction < 1 ? fraction : nil
                        self?.status = message
                    }
                }
                downloadProgress = nil

                for precision in precisions {
                    let report = try await engine.run(
                        modelPath: modelPath, precision: precision
                    ) { [weak self] phase in
                        Task { @MainActor in self?.status = Self.describe(phase) }
                    }
                    reports.append(report)
                }
                status = "Done — \(reports.count) run(s)"
            } catch {
                errorMessage = error.localizedDescription
                status = "Failed"
            }
            isRunning = false
        }
    }

    var combinedMarkdown: String {
        reports.map { $0.markdown() }.joined(separator: "\n\n")
    }

    private static func describe(_ phase: BenchmarkEngine.Phase) -> String {
        switch phase {
        case .idle: return "Idle"
        case .loading: return "Loading model…"
        case .warming(let s): return "Warming up \(s)…"
        case .running(let s, let p): return "Running \(s) (\(p.label))…"
        case .done: return "Finishing…"
        case .failed(let m): return "Failed: \(m)"
        }
    }
}

struct BenchmarkView: View {
    @StateObject private var model = BenchmarkViewModel()
    @State private var selection: Set<Precision> = [.fp16, .int8]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    precisionPicker
                    runButton
                    statusRow
                    if let error = model.errorMessage {
                        Text(error).font(.footnote).foregroundStyle(.red)
                    }
                    ForEach(model.reports, id: \.precision) { report in
                        ReportCard(report: report)
                    }
                    if !model.reports.isEmpty {
                        ShareLink(item: model.combinedMarkdown) {
                            Label("Share results (Markdown)", systemImage: "square.and.arrow.up")
                        }
                        .padding(.top, 4)
                    }
                }
                .padding()
            }
            .navigationTitle("GLiNER2 Bench")
        }
    }

    private var precisionPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Precision").font(.headline)
            ForEach(Precision.allCases, id: \.self) { precision in
                Toggle(precision.label, isOn: Binding(
                    get: { selection.contains(precision) },
                    set: { on in
                        if on { selection.insert(precision) } else { selection.remove(precision) }
                    }
                ))
                .disabled(model.isRunning)
            }
        }
    }

    private var runButton: some View {
        Button {
            model.run(precisions: Precision.allCases.filter { selection.contains($0) })
        } label: {
            Text(model.isRunning ? "Running…" : "Run benchmark")
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .disabled(model.isRunning || selection.isEmpty)
    }

    private var statusRow: some View {
        HStack(spacing: 10) {
            if model.isRunning { ProgressView() }
            if let progress = model.downloadProgress {
                ProgressView(value: progress).frame(width: 120)
            }
            Text(model.status).font(.subheadline).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Report card

private struct ReportCard: View {
    let report: BenchmarkReport

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(report.precision.label).font(.headline)

            grid([
                ("Load", "\(fmt(report.modelLoadMs)) ms"),
                ("MLX active", "\(fmt(report.mlxActiveMB)) MB"),
                ("MLX peak", "\(fmt(report.mlxPeakMB)) MB"),
                ("App RAM", "\(fmt(report.processResidentMB)) MB"),
            ])

            Divider()

            // Scenario table
            VStack(spacing: 0) {
                headerRow
                ForEach(report.scenarios) { s in
                    scenarioRow(s)
                    Divider()
                }
            }
            .font(.system(.footnote, design: .monospaced))

            Text("\(report.device) · iOS \(report.osVersion) · \(report.processorCount) cores")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding()
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
    }

    private var headerRow: some View {
        row(name: "scenario", p50: "p50", toks: "tok/s", mb: "MB", bold: true)
    }

    private func scenarioRow(_ s: ScenarioResult) -> some View {
        row(name: s.name,
            p50: fmt(s.p50Ms), toks: fmt(s.tokensPerSecond, 0), mb: fmt(s.peakMemoryMB, 0),
            bold: false)
    }

    private func row(name: String, p50: String, toks: String, mb: String, bold: Bool) -> some View {
        HStack(spacing: 6) {
            Text(name).frame(width: 110, alignment: .leading)
            Text(p50).frame(maxWidth: .infinity, alignment: .trailing)
            Text(toks).frame(maxWidth: .infinity, alignment: .trailing)
            Text(mb).frame(maxWidth: .infinity, alignment: .trailing)
        }
        .fontWeight(bold ? .bold : .regular)
        .padding(.vertical, 4)
    }

    private func grid(_ items: [(String, String)]) -> some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
            ForEach(items, id: \.0) { item in
                HStack {
                    Text(item.0).foregroundStyle(.secondary)
                    Spacer()
                    Text(item.1).bold()
                }
                .font(.footnote)
            }
        }
    }

    private func fmt(_ v: Double, _ places: Int = 1) -> String { String(format: "%.\(places)f", v) }
}
