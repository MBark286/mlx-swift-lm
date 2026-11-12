// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXGuidedGeneration

/// Byte-level tokenizer: token id = byte value, 255 = EOS.
private struct ByteTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.filter { $0 != 255 }.map(UInt8.init), encoding: .utf8) ?? ""
    }
    func convertTokenToId(_ token: String) -> Int? {
        if token == "<eos>" { return 255 }
        return token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
    }
    func convertIdToToken(_ id: Int) -> String? {
        if id == 255 { return "<eos>" }
        return id >= 0 && id < 255 ? String(UnicodeScalar(UInt8(id))) : nil
    }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Forced-completion sampling tests for ``GuidedGenerationLoop/applyMaskAndSample``.
@Suite
struct ForcedCompletionSamplingTests {

    @Test("Closing bias overrides model logit, selecting quote over continuation token")
    func closingBiasSelectsQuoteOverContinuation() {
        // 'A' (65) has higher raw logit than '"' (34),
        // but closing bias on '"' should flip the result.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[65] = 20.0  // 'A' continuation token, high logit
        floats[34] = 1.0  // '"' closing token, low logit
        let logits = MLXArray(floats)

        // Mask allowing both tokens
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[65 / 32] |= (1 << (65 % 32))
        maskWords[34 / 32] |= (1 << (34 % 32))

        // Closing bias: +100 on '"'
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[34] = 100.0
        let closingBias = MLXArray(biasFloats)

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...],
            maskArray: maskArray,
            closingBias: closingBias)
        #expect(result == 34)  // " wins due to bias despite lower model logit
    }

    @Test("The closing bias picks the quote over a digit")
    func closingBiasClosesStringInsteadOfDigit() {
        // Inside a string the grammar allows a digit ('0'=48) and the quote.
        // Digits get no closing bias, so the quote wins.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[48] = 20.0  // '0', high logit
        floats[34] = 1.0  // '"', low logit
        let logits = MLXArray(floats)

        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[48 / 32] |= (1 << (48 % 32))
        maskWords[34 / 32] |= (1 << (34 % 32))

        let closingBias = ClosingTokenBias.compute(
            tokenizer: ByteTokenizer(), eosTokenId: nil)

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...],
            maskArray: maskArray,
            closingBias: closingBias)
        #expect(result == 34)
    }

    @Test("The hard-zone bias picks the bracket over a digit")
    func hardZoneClosesArrayInsteadOfDigit() {
        // After `[1, 7` the grammar allows digits, ',' and ']'. Digits are
        // not closers, so the hard-zone bias leaves only ']'.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[55] = 20.0  // '7'
        floats[44] = 10.0  // ','
        floats[93] = 1.0  // ']'
        let logits = MLXArray(floats)

        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        for id in [55, 44, 93] { maskWords[id / 32] |= (1 << (id % 32)) }

        let closing = ClosingTokenBias.compute(tokenizer: ByteTokenizer(), eosTokenId: nil)
        let hardBias = which(closing .> 0, Float32(0.0), Float32(-10000.0))

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...],
            maskArray: maskArray,
            closingBias: hardBias)
        #expect(result == 93)
    }

    @Test("Closing bias has no effect when biased tokens are masked out by grammar")
    func closingBiasIgnoredWhenTokensMaskedOut() {
        // Only continuation tokens ('A'=65 and 'B'=66) are allowed.
        // '"' (34) has closing bias but is masked out -- must not be selected.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[65] = 20.0  // 'A'
        floats[66] = 10.0  // 'B'
        floats[34] = 1.0  // '"' -- will be masked out
        let logits = MLXArray(floats)

        // Mask allowing only 'A' and 'B'
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[65 / 32] |= (1 << (65 % 32))
        maskWords[66 / 32] |= (1 << (66 % 32))

        // Closing bias: +100 on '"'
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[34] = 100.0
        let closingBias = MLXArray(biasFloats)

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...],
            maskArray: maskArray,
            closingBias: closingBias)
        #expect(result == 65)  // 'A' wins -- highest logit among allowed tokens
    }

    @Test("Whitespace bias suppresses whitespace token, argmax selects non-whitespace")
    func whitespaceBiasSuppressesWhitespaceToken() {
        // Space (32) has the highest raw logit, but whitespace bias should
        // push it below 'A' (65) so the non-whitespace token wins.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[32] = 30.0  // space -- highest raw logit
        floats[65] = 10.0  // 'A' -- lower raw logit
        let logits = MLXArray(floats)

        // Mask allowing both tokens
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[32 / 32] |= (1 << (32 % 32))  // space
        maskWords[65 / 32] |= (1 << (65 % 32))  // 'A'

        // Whitespace bias: -200 on space token
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[32] = -200.0
        let whitespaceBias = MLXArray(biasFloats)

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...],
            maskArray: maskArray,
            closingBias: whitespaceBias)
        #expect(result == 65)  // 'A' wins -- whitespace bias suppressed space
    }

    @Test(
        "When all grammar-allowed tokens are whitespace, bias reduces but does not block selection"
    )
    func whitespaceBiasDoesNotBlockWhenAllAllowedAreWhitespace() {
        // Grammar allows only space (32) and tab (9). Both are whitespace.
        // Whitespace bias makes them negative, but they should still be
        // selectable (least-negative beats -inf on disallowed tokens).
        var floats = [Float](repeating: 0.0, count: 256)
        floats[32] = 10.0  // space -- higher raw logit
        floats[9] = 5.0  // tab -- lower raw logit
        let logits = MLXArray(floats)

        // Mask allowing only space and tab
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[32 / 32] |= (1 << (32 % 32))  // space
        maskWords[9 / 32] |= (1 << (9 % 32))  // tab

        // Whitespace bias: -200 on both whitespace tokens
        var biasFloats = [Float](repeating: 0.0, count: 256)
        biasFloats[32] = -200.0
        biasFloats[9] = -200.0
        let whitespaceBias = MLXArray(biasFloats)

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...],
            maskArray: maskArray,
            closingBias: whitespaceBias)
        // Space has logit 10 + bias -200 = -190; tab has 5 + -200 = -195.
        // All other tokens are -inf (masked out). Space wins as least-negative.
        #expect(result == 32)
    }

    @Test("nil closingBias selects highest allowed logit")
    func nilClosingBiasMatchesOriginalBehavior() {
        // Without closing bias, argmax of allowed tokens wins.
        var floats = [Float](repeating: 0.0, count: 256)
        floats[65] = 20.0  // 'A' -- highest
        floats[34] = 15.0  // '"' -- second highest
        let logits = MLXArray(floats)

        // Mask allowing both
        var maskWords = [UInt32](repeating: 0, count: 256 / 32)
        maskWords[65 / 32] |= (1 << (65 % 32))
        maskWords[34 / 32] |= (1 << (34 % 32))

        let maskArray = maskWords.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray(
                $0.baseAddress!, maskBitCount: 256, totalCount: 256)
        }
        let result = GuidedGenerationLoop.applyMaskAndSample(
            logits: logits[.newAxis, .newAxis, 0...],
            maskArray: maskArray,
            closingBias: nil)
        #expect(result == 65)  // 'A' wins -- no bias applied
    }
}

/// Scores the next byte from the bytes emitted so far: `next` gets 10, `fallback` gets 9.
private final class ScriptedModel: Module, LanguageModel {
    let promptLength: Int
    let next: (Int) -> (next: UInt8, fallback: UInt8?)
    var seen = 0

    init(promptLength: Int, next: @escaping (Int) -> (next: UInt8, fallback: UInt8?)) {
        self.promptLength = promptLength
        self.next = next
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let count = input.tokens.size
        seen += count
        let choice = next(seen - promptLength)
        var row = (0 ..< 256).map { -Float($0) / 1000 }
        row[Int(choice.next)] = 10
        if let fallback = choice.fallback { row[Int(fallback)] = 9 }
        let rows = Array(repeating: row, count: count).flatMap { $0 }
        return LMOutput(logits: MLXArray(rows).reshaped(1, count, 256))
    }
}

private struct UnusedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("the test supplies the prepared input")
    }
}

/// Closing-zone tests on the whole loop, with a scripted model and a real grammar.
@Suite(.serialized)
struct ForcedCompletionLoopTests {

    private func generate(
        grammar: String, maxTokens: Int, completionReserve: Int, hardReserve: Int = 0,
        next: @escaping (Int) -> (next: UInt8, fallback: UInt8?)
    ) throws -> String {
        try Stream.withNewDefaultStream(device: .cpu) {
            let tokenizer = ByteTokenizer()
            let vocabulary = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
            let grammarTokenizer = try GrammarTokenizer(
                vocab: vocabulary, vocabType: .byteFallback, eosTokenId: 255)
            let constraint = try GrammarConstraint(
                tokenizer: grammarTokenizer, grammar: grammar, fastForward: true,
                hostTokenizer: tokenizer)
            let context = ModelContext(
                configuration: ModelConfiguration(id: "scripted"),
                model: ScriptedModel(promptLength: 1, next: next),
                processor: UnusedProcessor(), tokenizer: tokenizer)
            var output = ""
            try GuidedGenerationLoop.run(
                input: LMInput(tokens: MLXArray([Int32(1)])), context: context,
                constraint: constraint, maxTokens: maxTokens, vocabSize: 256,
                completionReserve: completionReserve, hardReserve: hardReserve,
                closingBias: ClosingTokenBias.compute(tokenizer: tokenizer, eosTokenId: 255)
            ) {
                output += $0
                return true
            }
            return output
        }
    }

    @Test("A number in progress when the soft zone starts is finished, then the array closes")
    func numberInProgressFinishes() throws {
        // The model counts "[11,12,13,...". The soft zone starts at byte 29,
        // between the two digits of "20".
        let script = Array("[" + (11 ... 40).map(String.init).joined(separator: ",") + "]").map {
            $0.asciiValue!
        }
        let output = try generate(
            grammar: #"root ::= "[" [0-9]+ ("," [0-9]+)* "]""#,
            maxTokens: 40, completionReserve: 11
        ) { index in (script[min(index, script.count - 1)], nil) }
        #expect(output == "[11,12,13,14,15,16,17,18,19,20]")
    }

    @Test("Inside a string, a digit that is only the model's second choice does not pass")
    func stringClosesWithoutDigits() throws {
        let output = try generate(
            grammar: #"root ::= "{\"a\":\"" [a-z0-9]* "\"}""#,
            maxTokens: 40, completionReserve: 10, hardReserve: 4
        ) { _ in (UInt8(ascii: "x"), UInt8(ascii: "7")) }
        #expect(output == #"{"a":""# + String(repeating: "x", count: 24) + #""}"#)
    }

    @Test("A model that only picks digits is stopped after the run limit, and the JSON closes")
    func digitRunStopsAtLimit() throws {
        let output = try generate(
            grammar: #"root ::= "{\"a\":\"" [a-z0-9]* "\"}""#,
            maxTokens: 60, completionReserve: 30, hardReserve: 10
        ) { index in (index < 28 ? UInt8(ascii: "x") : UInt8(ascii: "7"), nil) }
        let digits = String(repeating: "7", count: GuidedGenerationLoop.maxZoneDigitRun)
        #expect(output == #"{"a":""# + String(repeating: "x", count: 22) + digits + #""}"#)
    }
}
