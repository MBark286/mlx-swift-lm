// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXEmbedders

// Two tiny random ModernBERT checkpoints with the states transformers 5.17 computes for them
// (eager attention, float32, CPU), in expected.safetensors:
// - tiny-silu: SiLU, keys without prefix and a transformers 4 config.json, as the Granite 97M.
// - tiny-gelu: GELU, saved as ModernBertForMaskedLM (keys under `model.` plus the head) with a
//   transformers 5 config.json, as answerdotai/ModernBERT-base.
// Cases: "long" [1, 24] (wider than the local window of 4 + 1 + 4), "short" [1, 5], and "batch"
// [2, 16], right-padded (the second row has 9 real tokens).

private struct FixtureTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        TestTokenizer()
    }
}

private func fixture(_ name: String) throws -> URL {
    try #require(
        Bundle.module.url(
            forResource: "config", withExtension: "json", subdirectory: "ModernBert/\(name)")
    )
    .deletingLastPathComponent()
}

private func relNorm(_ a: MLXArray, _ b: MLXArray) -> Float {
    let difference = (a - b).square().sum().sqrt()
    return (difference / b.square().sum().sqrt()).item(Float.self)
}

private func decode(_ json: String) throws -> ModernBertConfiguration {
    try JSONDecoder().decode(ModernBertConfiguration.self, from: Data(json.utf8))
}

private let tinyConfig = """
    "model_type": "modernbert", "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
    "num_hidden_layers": 4, "num_attention_heads": 2
    """

struct ModernBertTests {

    @Test(
        "Every layer matches transformers on the tiny checkpoints",
        arguments: ["tiny-silu", "tiny-gelu"])
    func layerParity(name: String) async throws {
        let directory = try fixture(name)
        let context = try await EmbedderModelFactory.shared.load(
            from: directory, using: FixtureTokenizerLoader())
        let model = try #require(context.model as? ModernBertModel)
        let expected = try loadArrays(url: directory.appending(path: "expected.safetensors"))
        let names =
            ["emb"]
            + (0 ..< model.configuration.hiddenLayers).map { String(format: "layer.%02d", $0) }
            + ["final"]

        try Device.withDefaultDevice(.cpu) {
            for test in ["long", "short", "batch"] {
                let ids = try #require(expected["\(test).input_ids"])
                let mask = try #require(expected["\(test).attention_mask"])
                let cls = try #require(expected["\(test).cls"])
                // Padded positions are not compared: each side fills them its own way. Real
                // tokens never see padded keys.
                let real = mask.asType(.float32)[.ellipsis, .newAxis]
                // "long" and "short" have no padding, so they also run without a mask.
                let runs: [(String, MLXArray?)] =
                    test == "batch" ? [("mask", mask)] : [("mask", mask), ("no mask", nil)]

                for (label, attentionMask) in runs {
                    let states = model.layerStates(ids, attentionMask: attentionMask)
                    #expect(states.count == names.count)
                    for (state, stateName) in zip(states, names) {
                        let reference = try #require(expected["\(test).\(stateName)"])
                        #expect(
                            relNorm(state * real, reference * real) <= 1e-5,
                            "\(test), \(label): \(stateName)")
                    }

                    let output = model(
                        ids, positionIds: nil, tokenTypeIds: nil, attentionMask: attentionMask)
                    let pooled = context.pooling(output, normalize: true)
                    #expect(relNorm(pooled, cls) <= 1e-5, "\(test), \(label): cls")
                }
            }
        }
    }

    @Test("A transformers 5 config is read from rope_parameters and layer_types")
    func transformers5Config() throws {
        // Values that differ from the defaults and from the i % 3 pattern.
        let configuration = try decode(
            """
            {\(tinyConfig),
             "layer_types": ["full_attention", "full_attention", "sliding_attention", "full_attention"],
             "rope_parameters": {
                "full_attention": {"rope_type": "default", "rope_theta": 150000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 20000.0}}}
            """)
        #expect(configuration.globalRopeTheta == 150_000)
        #expect(configuration.localRopeTheta == 20_000)
        #expect(configuration.globalLayers == [true, true, false, true])
    }

    @Test("A transformers 4 config, and the defaults")
    func transformers4Config() throws {
        let legacy = try decode(
            """
            {\(tinyConfig), "global_rope_theta": 150000.0, "local_rope_theta": 160000.0,
             "global_attn_every_n_layers": 2, "hidden_activation": "silu", "rope_scaling": null}
            """)
        #expect(legacy.globalRopeTheta == 150_000)
        #expect(legacy.localRopeTheta == 160_000)
        #expect(legacy.globalLayers == [true, false, true, false])
        #expect(legacy.hiddenActivation == .silu)

        let defaults = try decode("{\(tinyConfig)}")
        #expect(defaults.globalRopeTheta == 160_000)
        #expect(defaults.localRopeTheta == 10_000)
        #expect(defaults.globalLayers == [true, false, false, true])
        #expect(defaults.hiddenActivation == .gelu)
        #expect(defaults.localAttention == 128)
    }

    @Test(
        "Configs the port cannot run fail to decode",
        arguments: [
            #""rope_scaling": {"type": "linear", "factor": 2.0}"#,
            #""rope_parameters": {"full_attention": {"rope_type": "linear", "factor": 2.0}}"#,
            #""rope_parameters": {"sliding_attention": {"type": "linear", "factor": 2.0}}"#,
            #""hidden_activation": "gelu_pytorch_tanh""#,
            #""local_attention": 127"#,
            #""layer_types": ["full_attention", "full_attention", "sliding_attention", "full_attention"], "global_attn_every_n_layers": 3"#,
            #""layer_types": ["full_attention"]"#,
        ])
    func rejectedConfig(entry: String) {
        #expect(throws: DecodingError.self) { try decode("{\(tinyConfig), \(entry)}") }
    }
}
