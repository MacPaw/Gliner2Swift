// HubLoadingTests.swift
// Tests for Hub-based model downloading and loading via swift-transformers Hub package.
//
// These tests verify:
// 1. downloadModelDirectory() works with HuggingFace Hub
// 2. GLiNER2.fromPretrained() loads correctly from Hub
// 3. Hub-loaded model produces identical results to local path loading
// 4. Parity with Python-generated fixtures (same tests as RawWeightsParityTests but via Hub)

import XCTest
import Foundation
import Metal
@testable import GLiNER2Swift
import MLX
import MLXNN
import Hub

final class HubLoadingTests: XCTestCase {

    static let repoId = "fastino/gliner2-base-v1"
    static let localWeightsPath: String = {
        if let envPath = ProcessInfo.processInfo.environment["GLINER2_WEIGHTS_PATH"] {
            return envPath
        }
        return URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights").path
    }()

    // Path to raw_weights fixtures (same fixtures used by RawWeightsParityTests)
    static let fixturesPath: String = {
        let testFile = URL(fileURLWithPath: #file)
        return testFile.deletingLastPathComponent()
            .appendingPathComponent("Fixtures/raw_weights")
            .path
    }()

    // MARK: - Helpers

    private func skipIfNoGPU() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU not available")
        }
    }

    private func skipIfNoLocalModel() throws {
        guard FileManager.default.fileExists(atPath: Self.localWeightsPath + "/model.safetensors") else {
            throw XCTSkip("Local model not available at \(Self.localWeightsPath)")
        }
    }

    private func skipIfNoFixtures() throws {
        guard FileManager.default.fileExists(atPath: Self.fixturesPath) else {
            throw XCTSkip("Fixtures not available at \(Self.fixturesPath)")
        }
    }

    private func loadFixtureJSON(_ name: String) throws -> [String: Any] {
        let url = URL(fileURLWithPath: Self.fixturesPath).appendingPathComponent("\(name).json")
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "HubLoadingTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to parse \(name).json"])
        }
        return json
    }

    private func loadHubModel() async throws -> GLiNER2 {
        do {
            return try await GLiNER2.fromPretrained(Self.repoId)
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain {
                throw XCTSkip("Network unavailable: \(error.localizedDescription)")
            }
            throw error
        }
    }

    // MARK: - Hub Download Tests

    func testDownloadModelDirectory() async throws {
        try skipIfNoGPU()

        let modelDir: URL
        do {
            modelDir = try await downloadModelDirectory(repoId: Self.repoId)
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain {
                throw XCTSkip("Network unavailable: \(error.localizedDescription)")
            }
            throw error
        }

        // Verify returned URL is a valid directory
        var isDir: ObjCBool = false
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: modelDir.path, isDirectory: &isDir),
            "Model directory should exist"
        )
        XCTAssertTrue(isDir.boolValue, "Should be a directory")

        // Verify expected files exist
        for filename in ["config.json", "model.safetensors", "tokenizer.json"] {
            let path = modelDir.appendingPathComponent(filename)
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: path.path),
                "\(filename) should exist in downloaded directory"
            )
        }
    }

    func testHubCacheLocation() async throws {
        try skipIfNoGPU()

        // First download to populate cache
        do {
            _ = try await downloadModelDirectory(repoId: Self.repoId)
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain {
                throw XCTSkip("Network unavailable: \(error.localizedDescription)")
            }
            throw error
        }

        // Verify files are in the shared HF cache
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let repoDir = homeDir.appendingPathComponent(
            ".cache/huggingface/hub/models--fastino--gliner2-base-v1")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: repoDir.path),
            "Model should be cached at ~/.cache/huggingface/hub/models--fastino--gliner2-base-v1/"
        )

        // Second call should return immediately (cached)
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await downloadModelDirectory(repoId: Self.repoId)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        XCTAssertLessThan(elapsed, 5.0, "Cached download should return quickly (took \(elapsed)s)")
    }

    func testOfflineFallbackUsesCache() async throws {
        try skipIfNoGPU()

        // First, ensure model is cached by downloading normally
        do {
            _ = try await downloadModelDirectory(repoId: Self.repoId)
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain {
                throw XCTSkip("Network unavailable — cannot populate cache for offline test")
            }
            throw error
        }

        // Simulate offline with HubApi in explicit offline mode.
        // downloadModelDirectory catches the offlineModeError thrown by Hub
        // and falls back to localRepoLocation.
        let offlineHub = HubApi(useOfflineMode: true)
        let modelDir: URL
        do {
            modelDir = try await downloadModelDirectory(repoId: Self.repoId, hub: offlineHub)
        } catch {
            XCTFail("Offline fallback should use cached directory, but got error: \(error)")
            return
        }

        var isDir: ObjCBool = false
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: modelDir.path, isDirectory: &isDir),
            "Offline fallback should return a valid directory"
        )
    }

    func testProgressHandlerCalled() async throws {
        try skipIfNoGPU()

        let progressCallCount = ProgressCounter()

        do {
            _ = try await downloadModelDirectory(repoId: Self.repoId) { progress in
                progressCallCount.increment()
            }
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain {
                throw XCTSkip("Network unavailable: \(error.localizedDescription)")
            }
            throw error
        }

        print("Progress handler was called \(progressCallCount.count) time(s)")
    }

    // MARK: - Model Loading Tests

    func testFromPretrainedViaHubLoadsSuccessfully() async throws {
        try skipIfNoGPU()

        let model = try await loadHubModel()

        // Verify classifier weights are loaded (not random zeros)
        let classifierWeight = model.model.classifier.layers[0] as! Linear
        let weightSum = MLX.sum(MLX.abs(classifierWeight.weight))
        MLX.eval(weightSum)
        XCTAssertGreaterThan(Float(weightSum.item(Float32.self)), 0.0,
            "Classifier weights should not be all zeros after loading from Hub")
    }

    func testFromPretrainedLocalPathStillWorks() async throws {
        try skipIfNoGPU()
        try skipIfNoLocalModel()

        let model = try await GLiNER2.fromPretrained(Self.localWeightsPath)

        let result = model.extractEntities(
            text: "Elon Musk founded SpaceX.",
            entityTypes: ["person", "company"],
            includeConfidence: true,
            includeSpans: true
        )

        guard let entities = result["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities' key in result. Got: \(result)")
            return
        }

        let persons = entities["person"] as? [[String: Any]] ?? []
        XCTAssertFalse(persons.isEmpty, "Local path loading should detect person entities")

        let companies = entities["company"] as? [[String: Any]] ?? []
        XCTAssertFalse(companies.isEmpty, "Local path loading should detect company entities")

        print("Local path result: \(result)")
    }

    // MARK: - Hub vs Local Parity

    func testHubMatchesLocalPath() async throws {
        try skipIfNoGPU()
        try skipIfNoLocalModel()

        let hubModel = try await loadHubModel()
        let localModel = try await GLiNER2.fromPretrained(Self.localWeightsPath)

        let text = "Tim Cook is CEO of Apple."
        let entityTypes = ["person", "company"]

        let hubResult = hubModel.extractEntities(
            text: text, entityTypes: entityTypes,
            includeConfidence: true, includeSpans: true
        )
        let localResult = localModel.extractEntities(
            text: text, entityTypes: entityTypes,
            includeConfidence: true, includeSpans: true
        )

        guard let hubEntities = hubResult["entities"] as? [String: Any],
              let localEntities = localResult["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities' key. Hub: \(hubResult), Local: \(localResult)")
            return
        }

        for entityType in entityTypes {
            assertEntityListMatch(
                hubEntities[entityType], localEntities[entityType],
                entityType: entityType
            )
        }
    }

    // MARK: - Entity Parity (Hub vs Python fixtures)

    func testHubEntityBasicParity() async throws {
        try skipIfNoGPU()
        try skipIfNoFixtures()

        let model = try await loadHubModel()
        let expected = try loadFixtureJSON("entity_basic_result")

        let result = model.extractEntities(
            text: "Tim Cook is CEO of Apple.",
            entityTypes: ["person", "company"],
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        guard let expectedEntities = expected["entities"] as? [String: Any],
              let resultEntities = result["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities' key in result or expected")
            return
        }

        assertEntityListMatch(
            resultEntities["person"], expectedEntities["person"],
            entityType: "person"
        )
        assertEntityListMatch(
            resultEntities["company"], expectedEntities["company"],
            entityType: "company"
        )
    }

    func testHubEntityMultiParity() async throws {
        try skipIfNoGPU()
        try skipIfNoFixtures()

        let model = try await loadHubModel()
        let expected = try loadFixtureJSON("entity_multi_result")

        let result = model.extractEntities(
            text: "John and Jane work at Google in Mountain View.",
            entityTypes: ["person", "organization", "location"],
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        guard let expectedEntities = expected["entities"] as? [String: Any],
              let resultEntities = result["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities' key")
            return
        }

        for entityType in ["person", "organization", "location"] {
            assertEntityListMatch(
                resultEntities[entityType], expectedEntities[entityType],
                entityType: entityType
            )
        }
    }

    func testHubEntityNoMatchParity() async throws {
        try skipIfNoGPU()
        try skipIfNoFixtures()

        let model = try await loadHubModel()
        let expected = try loadFixtureJSON("entity_no_match_result")

        let result = model.extractEntities(
            text: "The weather is nice today.",
            entityTypes: ["person", "company"],
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        guard let expectedEntities = expected["entities"] as? [String: Any],
              let resultEntities = result["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities' key")
            return
        }

        for entityType in ["person", "company"] {
            let expectedList = expectedEntities[entityType] as? [Any] ?? []
            let resultList = resultEntities[entityType] as? [Any] ?? []
            XCTAssertEqual(resultList.count, expectedList.count,
                "Entity type '\(entityType)' count mismatch: got \(resultList.count), expected \(expectedList.count)")
        }
    }

    // MARK: - Classification Parity (Hub vs Python fixtures)

    func testHubClassifySentimentPositiveParity() async throws {
        try skipIfNoGPU()
        try skipIfNoFixtures()

        let model = try await loadHubModel()
        let expected = try loadFixtureJSON("classify_sentiment_positive_result")

        let result = model.classifyText(
            text: "Great product! I love it.",
            task: "sentiment",
            labels: ["positive", "negative", "neutral"],
            threshold: 0.5,
            includeConfidence: true
        )

        assertClassificationMatch(result, expected, task: "sentiment")
    }

    func testHubClassifySentimentNegativeParity() async throws {
        try skipIfNoGPU()
        try skipIfNoFixtures()

        let model = try await loadHubModel()
        let expected = try loadFixtureJSON("classify_sentiment_negative_result")

        let result = model.classifyText(
            text: "Terrible service. Very disappointed.",
            task: "sentiment",
            labels: ["positive", "negative", "neutral"],
            threshold: 0.5,
            includeConfidence: true
        )

        assertClassificationMatch(result, expected, task: "sentiment")
    }

    // MARK: - Structure Extraction Parity (Hub vs Python fixtures)

    func testHubStructPersonParity() async throws {
        try skipIfNoGPU()
        try skipIfNoFixtures()

        let model = try await loadHubModel()
        let expected = try loadFixtureJSON("struct_person_result")

        let schema = model.createSchema()
            .structure("person_info")
            .field("name")
            .field("age")
            .field("location")
            .done()

        let result = model.extract(
            text: "John Smith is 35 years old and lives in New York.",
            schema: schema,
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        assertStructureMatch(result, expected, structureName: "person_info",
                             fields: ["name", "age", "location"])
    }

    func testHubStructProductParity() async throws {
        try skipIfNoGPU()
        try skipIfNoFixtures()

        let model = try await loadHubModel()
        let expected = try loadFixtureJSON("struct_product_result")

        let schema = model.createSchema()
            .structure("product")
            .field("name")
            .field("price")
            .field("manufacturer")
            .done()

        let result = model.extract(
            text: "iPhone 15 Pro costs $999 and is made by Apple.",
            schema: schema,
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        assertStructureMatch(result, expected, structureName: "product",
                             fields: ["name", "price", "manufacturer"])
    }

    // MARK: - Assertion Helpers

    private func assertEntityListMatch(_ result: Any?, _ expected: Any?, entityType: String) {
        guard let expectedList = expected as? [[String: Any]] else {
            XCTFail("Expected list missing for entity type '\(entityType)'")
            return
        }

        guard let resultList = result as? [[String: Any]] else {
            if let resultStrList = result as? [String] {
                XCTAssertEqual(resultStrList.count, expectedList.count,
                    "Entity '\(entityType)' count mismatch")
                return
            }
            if let resultEmpty = result as? [Any], resultEmpty.isEmpty, expectedList.isEmpty {
                return
            }
            XCTFail("Result type mismatch for entity type '\(entityType)': \(type(of: result))")
            return
        }

        XCTAssertEqual(resultList.count, expectedList.count,
            "Entity '\(entityType)' count: got \(resultList.count), expected \(expectedList.count)")

        for (i, expectedEntity) in expectedList.enumerated() {
            guard i < resultList.count else { break }
            let resultEntity = resultList[i]

            let expectedText = expectedEntity["text"] as? String ?? ""
            let resultText = resultEntity["text"] as? String ?? ""
            XCTAssertEqual(resultText, expectedText,
                "Entity '\(entityType)' [\(i)] text: got '\(resultText)', expected '\(expectedText)'")

            if let expectedStart = expectedEntity["start"] as? Int,
               let resultStart = resultEntity["start"] as? Int {
                XCTAssertEqual(resultStart, expectedStart,
                    "Entity '\(entityType)' [\(i)] start: got \(resultStart), expected \(expectedStart)")
            }

            if let expectedEnd = expectedEntity["end"] as? Int,
               let resultEnd = resultEntity["end"] as? Int {
                XCTAssertEqual(resultEnd, expectedEnd,
                    "Entity '\(entityType)' [\(i)] end: got \(resultEnd), expected \(expectedEnd)")
            }
        }
    }

    private func assertClassificationMatch(_ result: [String: Any], _ expected: [String: Any], task: String) {
        guard let expectedTaskResult = expected[task] as? [String: Any],
              let expectedLabel = expectedTaskResult["label"] as? String else {
            XCTFail("Expected classification result missing for task '\(task)'")
            return
        }

        if let resultTaskResult = result[task] as? [String: Any],
           let resultLabel = resultTaskResult["label"] as? String {
            XCTAssertEqual(resultLabel, expectedLabel,
                "Classification '\(task)' label: got '\(resultLabel)', expected '\(expectedLabel)'")
        } else if let resultLabel = result[task] as? String {
            XCTAssertEqual(resultLabel, expectedLabel,
                "Classification '\(task)' label: got '\(resultLabel)', expected '\(expectedLabel)'")
        } else {
            XCTFail("Classification result type mismatch for task '\(task)': \(result)")
        }
    }

    private func assertStructureMatch(_ result: [String: Any], _ expected: [String: Any],
                                       structureName: String, fields: [String]) {
        guard let expectedInstances = expected[structureName] as? [[String: Any]] else {
            XCTFail("Expected structure instances missing for '\(structureName)'")
            return
        }

        guard let resultInstances = result[structureName] as? [[String: Any]] else {
            XCTFail("Result structure instances missing for '\(structureName)'. Got: \(result)")
            return
        }

        XCTAssertEqual(resultInstances.count, expectedInstances.count,
            "Structure '\(structureName)' instance count: got \(resultInstances.count), expected \(expectedInstances.count)")

        for (i, expectedInst) in expectedInstances.enumerated() {
            guard i < resultInstances.count else { break }
            let resultInst = resultInstances[i]

            for field in fields {
                guard let expectedFieldList = expectedInst[field] as? [[String: Any]] else { continue }
                guard let resultFieldList = resultInst[field] as? [[String: Any]] else {
                    XCTFail("Structure '\(structureName)' instance \(i) field '\(field)' missing in result")
                    continue
                }

                XCTAssertEqual(resultFieldList.count, expectedFieldList.count,
                    "Structure '\(structureName)' [\(i)].\(field) count: got \(resultFieldList.count), expected \(expectedFieldList.count)")

                for (j, expectedItem) in expectedFieldList.enumerated() {
                    guard j < resultFieldList.count else { break }
                    let resultItem = resultFieldList[j]

                    let expectedText = expectedItem["text"] as? String ?? ""
                    let resultText = resultItem["text"] as? String ?? ""
                    XCTAssertEqual(resultText, expectedText,
                        "Structure '\(structureName)' [\(i)].\(field)[\(j)] text: got '\(resultText)', expected '\(expectedText)'")
                }
            }
        }
    }
}

// MARK: - Helpers

/// Thread-safe counter for tracking progress handler calls.
private final class ProgressCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _count = 0

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return _count
    }

    func increment() {
        lock.lock()
        _count += 1
        lock.unlock()
    }
}