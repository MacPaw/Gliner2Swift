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
// UnigramTokenizer.swift
// SentencePiece Unigram tokenizer implementation for DeBERTa v3
//
// Implements the Unigram Language Model tokenization algorithm using
// Viterbi dynamic programming for optimal segmentation.
//
// Reference: https://arxiv.org/abs/1804.10959

import Foundation

// MARK: - Unigram Tokenizer

/// SentencePiece Unigram tokenizer for DeBERTa v3.
///
/// Uses Viterbi algorithm to find the globally optimal tokenization
/// that maximizes the total log-probability of the token sequence.
public final class UnigramTokenizer: @unchecked Sendable {

    // MARK: - Properties

    /// Vocabulary: token -> (id, score)
    private let vocab: [String: (id: Int, score: Float)]

    /// Reverse vocabulary: id -> token
    private let idToToken: [Int: String]

    /// Special tokens that should not be tokenized
    private let specialTokens: Set<String>

    /// Special token to ID mapping (for tokens from added_tokens)
    private let specialTokenToId: [String: Int]

    /// First characters of every special token.
    ///
    /// `preTokenize` used to test all ~15 special tokens with `hasPrefix` at every
    /// character of every word. One set lookup rejects the overwhelming majority of
    /// positions (in practice every special token starts with "[").
    private let specialTokenFirstChars: Set<Character>

    /// Upper bound on the length of any vocabulary token, in Characters.
    ///
    /// Viterbi's inner loop tries every substring up to this length; the bound was
    /// hardcoded at 50 while the real maximum is far smaller, so every position paid for
    /// substrings that could not possibly be in the vocabulary.
    private let maxPieceLength: Int

    /// Collapses whitespace runs and tabs/newlines, matching the leading `Replace` stage
    /// of the normalizer Python actually runs (see `normalize` below).
    private static let whitespaceRunPattern = try? NSRegularExpression(
        pattern: "\\s{2,}|[\\n\\r\\t]", options: [])

    /// Memoized `encode` results, keyed by the exact input string.
    ///
    /// The processor encodes one token at a time, and the schema half of the prompt is
    /// byte-identical on every call with the same schema — so without memoization every
    /// entity name, field name and (long) description string is re-run through Viterbi
    /// for every text and every inference. Natural-language word frequency is Zipfian, so
    /// the text half hits often too.
    ///
    /// Guarded because Phase 5.3 parallelizes per-text preprocessing; the lock costs far
    /// less than re-running the Viterbi lattice.
    private var encodeCache: [String: [Int]] = [:]
    private let encodeCacheLock = NSLock()

    /// Upper bound on cached entries, so a long-running process cannot grow unboundedly
    /// on adversarial or highly varied input.
    private static let encodeCacheLimit = 100_000

    /// Special token IDs
    public let padTokenId: Int
    public let clsTokenId: Int
    public let sepTokenId: Int
    public let unkTokenId: Int
    public let maskTokenId: Int

    /// GLiNER2 special token IDs
    public let sepStructId: Int
    public let sepTextId: Int
    public let pTokenId: Int
    public let cTokenId: Int
    public let eTokenId: Int
    public let rTokenId: Int
    public let lTokenId: Int
    public let exampleTokenId: Int
    public let outputTokenId: Int
    public let descriptionTokenId: Int

    /// Metaspace character (word boundary marker)
    public static let metaspace: Character = "\u{2581}"  // ▁
    public static let metaspaceString = "\u{2581}"

    /// Vocabulary size
    public var vocabSize: Int { vocab.count }

    // MARK: - Initialization

    /// Initialize from tokenizer.json file
    ///
    /// - Parameter tokenizerJsonUrl: URL to tokenizer.json
    public convenience init(tokenizerJsonUrl: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonUrl)
        try self.init(tokenizerJsonData: data)
    }

    /// Initialize from tokenizer.json data
    ///
    /// - Parameter tokenizerJsonData: Raw JSON data
    public init(tokenizerJsonData: Data) throws {
        guard let json = try JSONSerialization.jsonObject(with: tokenizerJsonData) as? [String: Any] else {
            throw TokenizerError.invalidFormat("Failed to parse tokenizer.json")
        }

        // Parse vocabulary from model.vocab
        guard let model = json["model"] as? [String: Any],
              let vocabArray = model["vocab"] as? [[Any]] else {
            throw TokenizerError.invalidFormat("Missing model.vocab in tokenizer.json")
        }

        var vocab: [String: (id: Int, score: Float)] = [:]
        var idToToken: [Int: String] = [:]
        var longestPiece = 1

        for (idx, entry) in vocabArray.enumerated() {
            guard entry.count >= 2,
                  let token = entry[0] as? String else {
                continue
            }

            // Score can be Int (0) or Double
            let score: Float
            if let doubleScore = entry[1] as? Double {
                score = Float(doubleScore)
            } else if let intScore = entry[1] as? Int {
                score = Float(intScore)
            } else {
                score = 0.0
            }

            vocab[token] = (id: idx, score: score)
            idToToken[idx] = token
            // UTF-8 count, not Character count: it is O(1) on a native Swift string,
            // whereas `count` breaks graphemes and cost ~40 ms of cold start across 128k
            // tokens. It can only over-estimate the Character length, so the Viterbi bound
            // stays correct — just marginally loose for multi-byte pieces.
            longestPiece = Swift.max(longestPiece, token.utf8.count)
        }

        // Parse added_tokens for special tokens
        var specialTokenSet: Set<String> = []
        var tokenToId: [String: Int] = [:]

        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for token in addedTokens {
                if let content = token["content"] as? String,
                   let id = token["id"] as? Int,
                   let isSpecial = token["special"] as? Bool, isSpecial {
                    specialTokenSet.insert(content)
                    tokenToId[content] = id
                }
            }
        }

        // Add reverse mapping for special tokens (needed for idToToken lookup)
        // This matches Python's convert_ids_to_tokens which includes special tokens
        for (content, id) in tokenToId {
            idToToken[id] = content
        }

        self.vocab = vocab
        self.specialTokenFirstChars = Set(specialTokenSet.compactMap { $0.first })
        self.maxPieceLength = longestPiece
        self.idToToken = idToToken
        self.specialTokens = specialTokenSet
        self.specialTokenToId = tokenToId

        // Set special token IDs (with fallbacks)
        self.padTokenId = tokenToId["[PAD]"] ?? vocab["[PAD]"]?.id ?? 0
        self.clsTokenId = tokenToId["[CLS]"] ?? vocab["[CLS]"]?.id ?? 1
        self.sepTokenId = tokenToId["[SEP]"] ?? vocab["[SEP]"]?.id ?? 2
        self.unkTokenId = tokenToId["[UNK]"] ?? vocab["[UNK]"]?.id ?? 3
        self.maskTokenId = tokenToId["[MASK]"] ?? vocab["[MASK]"]?.id ?? 128000

        // GLiNER2 special tokens
        self.sepStructId = tokenToId["[SEP_STRUCT]"] ?? vocab["[SEP_STRUCT]"]?.id ?? 128001
        self.sepTextId = tokenToId["[SEP_TEXT]"] ?? vocab["[SEP_TEXT]"]?.id ?? 128002
        self.pTokenId = tokenToId["[P]"] ?? vocab["[P]"]?.id ?? 128003
        self.cTokenId = tokenToId["[C]"] ?? vocab["[C]"]?.id ?? 128004
        self.eTokenId = tokenToId["[E]"] ?? vocab["[E]"]?.id ?? 128005
        self.rTokenId = tokenToId["[R]"] ?? vocab["[R]"]?.id ?? 128006
        self.lTokenId = tokenToId["[L]"] ?? vocab["[L]"]?.id ?? 128007
        self.exampleTokenId = tokenToId["[EXAMPLE]"] ?? vocab["[EXAMPLE]"]?.id ?? 128008
        self.outputTokenId = tokenToId["[OUTPUT]"] ?? vocab["[OUTPUT]"]?.id ?? 128009
        self.descriptionTokenId = tokenToId["[DESCRIPTION]"] ?? vocab["[DESCRIPTION]"]?.id ?? 128010
    }

    /// Apply the normalizer chain that Python's tokenizer actually runs.
    ///
    /// The parity target is `AutoTokenizer.from_pretrained(<repo>)`, which builds a fast
    /// `DebertaV2Tokenizer` whose normalizer is:
    ///
    ///     Sequence[ Replace(Regex("\s{2,}|[\n\r\t]"), " "), NFC(), Strip(right) ]
    ///
    /// Note this is NOT the chain declared in the model directory's tokenizer.json
    /// (`Strip -> Precompiled(charsmap) -> Replace`). That file is an artifact of weight
    /// conversion; transformers derives the tokenizer from `spm.model` instead and never
    /// applies the SentencePiece charsmap. The difference is observable: the charsmap maps
    /// fullwidth `Ａ` to `A` and composes decomposed accents, whereas Python's
    /// `normalizer.normalize_str("Ａ")` returns `"Ａ"` unchanged. Implementing the charsmap
    /// therefore moves Swift AWAY from Python parity — NFC is what matches.
    private func normalize(_ text: String) -> String {
        var result = text
        if let pattern = Self.whitespaceRunPattern {
            result = pattern.stringByReplacingMatches(
                in: result, options: [],
                range: NSRange(result.startIndex..., in: result),
                withTemplate: " ")
        }
        // NFC: canonical composition, e.g. "e" + U+0301 -> "é".
        result = result.precomposedStringWithCanonicalMapping
        // Strip(strip_left: false, strip_right: true)
        while let last = result.last, last.isWhitespace { result.removeLast() }
        return result
    }

    // MARK: - Tokenization

    /// Tokenize text into tokens
    ///
    /// - Parameter text: Input text
    /// - Returns: Array of token strings
    public func tokenize(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        // Apply the normalizer chain (whitespace collapse, NFC, right-strip). The leading
        // whitespace trim is retained from the previous behaviour: preTokenize splits on
        // whitespace and drops empties, so it cannot change the token stream.
        let normalized = normalize(text.trimmingCharacters(in: .whitespaces))
        guard !normalized.isEmpty else { return [] }

        // Pre-tokenize: split into words (whitespace only, matches Python Metaspace)
        let words = preTokenize(normalized)

        var tokens: [String] = []
        for word in words {
            // Check if it's a special token
            if specialTokens.contains(word) {
                tokens.append(word)
                continue
            }

            // Add metaspace for word-initial tokens (except first word after special)
            let wordWithMeta = Self.metaspaceString + word

            // Tokenize using Viterbi algorithm
            let wordTokens = viterbiTokenize(wordWithMeta)
            tokens.append(contentsOf: wordTokens)
        }

        return tokens
    }

    /// Encode text into token IDs
    ///
    /// - Parameter text: Input text
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        encodeCacheLock.lock()
        if let cached = encodeCache[text] {
            encodeCacheLock.unlock()
            return cached
        }
        encodeCacheLock.unlock()

        let ids = tokensToIds(tokenize(text))

        encodeCacheLock.lock()
        if encodeCache.count < Self.encodeCacheLimit {
            encodeCache[text] = ids
        }
        encodeCacheLock.unlock()
        return ids
    }

    /// Encode text with [CLS] and [SEP] tokens
    ///
    /// - Parameter text: Input text
    /// - Returns: Array of token IDs with special tokens
    public func encodeWithSpecialTokens(_ text: String) -> [Int] {
        var ids = [clsTokenId]
        ids.append(contentsOf: encode(text))
        ids.append(sepTokenId)
        return ids
    }

    /// Decode token IDs back to text
    ///
    /// - Parameter ids: Array of token IDs
    /// - Returns: Decoded text
    public func decode(_ ids: [Int]) -> String {
        let tokens = idsToTokens(ids)
        return tokensToText(tokens)
    }

    /// Convert tokens to IDs
    ///
    /// - Parameter tokens: Array of token strings
    /// - Returns: Array of token IDs
    public func tokensToIds(_ tokens: [String]) -> [Int] {
        tokens.map { token in
            // Check special tokens first (from added_tokens)
            if let specialId = specialTokenToId[token] {
                return specialId
            }
            // Then check regular vocab
            return vocab[token]?.id ?? unkTokenId
        }
    }

    /// Convert IDs to tokens
    ///
    /// - Parameter ids: Array of token IDs
    /// - Returns: Array of token strings
    public func idsToTokens(_ ids: [Int]) -> [String] {
        ids.compactMap { id in
            idToToken[id]
        }
    }

    /// Convert token to ID
    ///
    /// - Parameter token: Token string
    /// - Returns: Token ID or nil if not found
    public func tokenToId(_ token: String) -> Int? {
        vocab[token]?.id
    }

    /// Convert ID to token
    ///
    /// - Parameter id: Token ID
    /// - Returns: Token string or nil if not found
    public func idToToken(_ id: Int) -> String? {
        idToToken[id]
    }

    // MARK: - Pre-tokenization

    /// Split text into words while preserving special tokens.
    ///
    /// Matches Python's Metaspace pre-tokenizer behavior:
    /// - Split ONLY on whitespace (not punctuation)
    /// - Punctuation stays attached to the word
    /// - Special tokens (like [P], [E], etc.) are preserved as-is
    private func preTokenize(_ text: String) -> [String] {
        var words: [String] = []
        var currentWord = ""
        var i = text.startIndex

        while i < text.endIndex {
            // Check for special tokens at current position
            // The first-character test rejects nearly every position for the cost of one
            // set lookup, instead of ~15 `hasPrefix` calls per character.
            var foundSpecial = false
            if specialTokenFirstChars.contains(text[i]) {
                for special in specialTokens {
                    if text[i...].hasPrefix(special) {
                        // Save current word
                        if !currentWord.isEmpty {
                            words.append(currentWord)
                            currentWord = ""
                        }
                        // Add special token
                        words.append(special)
                        i = text.index(i, offsetBy: special.count)
                        foundSpecial = true
                        break
                    }
                }
            }

            if foundSpecial { continue }

            let char = text[i]

            // Split ONLY on whitespace (Metaspace behavior)
            // Punctuation stays attached to the word
            if char.isWhitespace {
                if !currentWord.isEmpty {
                    words.append(currentWord)
                    currentWord = ""
                }
            } else {
                // ALL other characters (including punctuation) stay with word
                currentWord.append(char)
            }

            i = text.index(after: i)
        }

        if !currentWord.isEmpty {
            words.append(currentWord)
        }

        return words
    }

    // MARK: - Viterbi Tokenization

    /// Tokenize a single word using Viterbi algorithm
    ///
    /// Finds the optimal tokenization that maximizes total log-probability.
    /// Time complexity: O(L²) where L = word length
    private func viterbiTokenize(_ word: String) -> [String] {
        let chars = Array(word)
        let n = chars.count

        guard n > 0 else { return [] }

        // Check if entire word is in vocabulary
        if let _ = vocab[word] {
            return [word]
        }

        // dp[i] = best score to reach position i
        var dp = [Float](repeating: -Float.infinity, count: n + 1)
        dp[0] = 0.0

        // parent[i] = (start_pos, token) that led to position i
        var parent: [(Int, String)?] = Array(repeating: nil, count: n + 1)

        // Forward pass: find best tokenization
        for i in 0..<n {
            guard dp[i] > -Float.infinity else { continue }

            // Try all possible tokens starting at position i
            for j in (i + 1)...(min(i + maxPieceLength, n)) {
                let substring = String(chars[i..<j])

                if let (_, score) = vocab[substring] {
                    let newScore = dp[i] + score
                    if newScore > dp[j] {
                        dp[j] = newScore
                        parent[j] = (i, substring)
                    }
                }
            }
        }

        // If no valid tokenization found, fall back to character-level
        if dp[n] == -Float.infinity {
            return fallbackTokenize(word)
        }

        // Backtrack to recover tokens
        var tokens: [String] = []
        var pos = n
        while pos > 0 {
            guard let (prevPos, token) = parent[pos] else {
                // This shouldn't happen if dp[n] > -inf
                return fallbackTokenize(word)
            }
            tokens.append(token)
            pos = prevPos
        }

        return tokens.reversed()
    }

    /// Fallback tokenization for unknown words
    ///
    /// Encodes each character as UTF-8 bytes mapped to control tokens
    private func fallbackTokenize(_ word: String) -> [String] {
        var tokens: [String] = []

        for char in word {
            // Check if single character is in vocab
            let charStr = String(char)
            if vocab[charStr] != nil {
                tokens.append(charStr)
            } else {
                // Encode as UTF-8 bytes -> control tokens
                let utf8Bytes = charStr.utf8
                for byte in utf8Bytes {
                    let controlToken = String(format: "<0x%02X>", byte)
                    tokens.append(controlToken)
                }
            }
        }

        return tokens
    }

    // MARK: - Decoding

    /// Convert tokens to text
    private func tokensToText(_ tokens: [String]) -> String {
        var result = ""

        for token in tokens {
            // Skip special tokens in output
            if specialTokens.contains(token) {
                continue
            }

            // Handle control bytes
            if token.hasPrefix("<0x") && token.hasSuffix(">") {
                if let byte = parseControlByte(token) {
                    result.append(Character(UnicodeScalar(byte)))
                }
                continue
            }

            // Replace metaspace with space
            let text = token.replacingOccurrences(of: Self.metaspaceString, with: " ")
            result.append(text)
        }

        // Trim leading space (first metaspace)
        if result.hasPrefix(" ") {
            result.removeFirst()
        }

        return result
    }

    /// Parse control byte token like <0xF0>
    private func parseControlByte(_ token: String) -> UInt8? {
        guard token.hasPrefix("<0x") && token.hasSuffix(">") else {
            return nil
        }
        let hexStr = String(token.dropFirst(3).dropLast(1))
        return UInt8(hexStr, radix: 16)
    }
}

// MARK: - Errors

public enum TokenizerError: Error, LocalizedError {
    case invalidFormat(String)
    case fileNotFound(String)
    case tokenNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .invalidFormat(let reason):
            return "Invalid tokenizer format: \(reason)"
        case .fileNotFound(let path):
            return "Tokenizer file not found: \(path)"
        case .tokenNotFound(let token):
            return "Token not found in vocabulary: \(token)"
        }
    }
}
