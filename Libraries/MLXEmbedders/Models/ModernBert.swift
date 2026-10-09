// Copyright © 2026 Apple Inc.

// ModernBERT encoder (transformers `modeling_modernbert.py`), for embedding checkpoints such as
// ibm-granite/granite-embedding-97m-multilingual-r2. Global layers attend to every token, local
// layers to `local_attention / 2` tokens on each side; both hide padded keys.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration for ``ModernBertModel``, decoded from a ModernBERT `config.json`.
///
/// Reads both the transformers 4 keys (`global_rope_theta`, `local_rope_theta`,
/// `global_attn_every_n_layers`) and the transformers 5 keys (`rope_parameters`, `layer_types`).
/// Missing keys take the transformers defaults. Configurations this port cannot run correctly
/// fail to decode: RoPE scaling, an unknown activation, an odd `local_attention`, and
/// `layer_types` that disagree with `global_attn_every_n_layers`.
public struct ModernBertConfiguration: Decodable, Sendable {

    /// The MLP activation (transformers `hidden_activation`).
    public enum Activation: String, Sendable {
        /// Exact GELU (transformers `gelu`), not the tanh approximation.
        case gelu
        case silu
    }

    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let hiddenActivation: Activation
    public let normEps: Float
    public let normBias: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool
    /// The full width of the local attention window; local layers see half of it on each side.
    public let localAttention: Int
    public let globalRopeTheta: Float
    public let localRopeTheta: Float
    /// For each layer, `true` when it attends to every token, `false` for the local window.
    public let globalLayers: [Bool]

    /// The size of each attention head.
    public var headDim: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case hiddenActivation = "hidden_activation"
        case normEps = "norm_eps"
        case normBias = "norm_bias"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case localAttention = "local_attention"
        case globalAttentionEveryNLayers = "global_attn_every_n_layers"
        case layerTypes = "layer_types"
        case globalRopeTheta = "global_rope_theta"
        case localRopeTheta = "local_rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?
        let ropeType: String?
        // Older configs name rope_type `type`; transformers still reads it.
        let type: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
            case type
        }

        var kind: String { ropeType ?? type ?? "default" }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 50368
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1152
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 22
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 12
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        normBias = try c.decodeIfPresent(Bool.self, forKey: .normBias) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        localAttention = try c.decodeIfPresent(Int.self, forKey: .localAttention) ?? 128

        let activation = try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "gelu"
        guard let hiddenActivation = Activation(rawValue: activation) else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenActivation, in: c,
                debugDescription:
                    "Unsupported hidden_activation '\(activation)': expected gelu or silu")
        }
        self.hiddenActivation = hiddenActivation

        guard attentionHeads > 0, hiddenSize % attentionHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .attentionHeads, in: c,
                debugDescription:
                    "hidden_size \(hiddenSize) is not divisible by num_attention_heads \(attentionHeads)"
            )
        }
        // transformers floors an odd value; no checkpoint uses one, so that path is not supported.
        guard localAttention > 0, localAttention % 2 == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .localAttention, in: c,
                debugDescription:
                    "local_attention must be a positive even number, got \(localAttention)")
        }

        // RoPE: rope_parameters (transformers 5), then the transformers 4 keys, then the defaults.
        if c.contains(.ropeScaling), try !c.decodeNil(forKey: .ropeScaling) {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeScaling, in: c, debugDescription: "RoPE scaling is not supported")
        }
        let ropeParameters =
            try c.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters) ?? [:]
        if let scaled = ropeParameters.first(where: { $0.value.kind != "default" }) {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeParameters, in: c,
                debugDescription:
                    "RoPE scaling is not supported: \(scaled.key) has rope_type '\(scaled.value.kind)'"
            )
        }
        globalRopeTheta =
            try ropeParameters["full_attention"]?.ropeTheta
            ?? c.decodeIfPresent(Float.self, forKey: .globalRopeTheta) ?? 160_000
        localRopeTheta =
            try ropeParameters["sliding_attention"]?.ropeTheta
            ?? c.decodeIfPresent(Float.self, forKey: .localRopeTheta) ?? 10_000

        // Layer i is global when i % global_attn_every_n_layers == 0, so layer 0 is global.
        // layer_types (transformers 5) is used when present, and must agree with
        // global_attn_every_n_layers when both are set.
        let every = try c.decodeIfPresent(Int.self, forKey: .globalAttentionEveryNLayers)
        guard (every ?? 3) > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .globalAttentionEveryNLayers, in: c,
                debugDescription: "global_attn_every_n_layers must be positive")
        }
        let fromPattern = (0 ..< hiddenLayers).map { $0 % (every ?? 3) == 0 }
        if let layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            let known = ["full_attention", "sliding_attention"]
            guard layerTypes.count == hiddenLayers, layerTypes.allSatisfy({ known.contains($0) })
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes, in: c,
                    debugDescription:
                        "layer_types must list \(hiddenLayers) entries of full_attention or sliding_attention"
                )
            }
            let fromTypes = layerTypes.map { $0 == "full_attention" }
            if every != nil, fromTypes != fromPattern {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes, in: c,
                    debugDescription:
                        "layer_types disagree with global_attn_every_n_layers \(every ?? 0)")
            }
            globalLayers = fromTypes
        } else {
            globalLayers = fromPattern
        }
    }
}

// MARK: - Embeddings

/// Token embeddings followed by a LayerNorm. Positions enter only through RoPE.
private final class ModernBertEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ c: ModernBertConfiguration) {
        _tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.normEps, bias: c.normBias)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        norm(tokEmbeddings(inputs))
    }
}

// MARK: - Attention

/// Bidirectional multi-head attention with RoPE; the base depends on the layer type.
private final class ModernBertAttention: Module {
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    let rope: RoPE
    let heads: Int
    let scale: Float

    init(_ c: ModernBertConfiguration, isGlobal: Bool) {
        heads = c.attentionHeads
        scale = pow(Float(c.headDim), -0.5)
        _wqkv.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: c.attentionBias)
        _wo.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.attentionBias)
        rope = RoPE(
            dimensions: c.headDim, traditional: false,
            base: isGlobal ? c.globalRopeTheta : c.localRopeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        // Wqkv packs q, k and v one after the other, each with its heads contiguous.
        let qkv = wqkv(x).split(parts: 3, axis: -1).map {
            $0.reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: rope(qkv[0]), keys: rope(qkv[1]), values: qkv[2], scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return wo(output)
    }
}

// MARK: - MLP

/// Gated MLP: `Wi` gives the activation input and the gate side by side.
private final class ModernBertMLP: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    let activation: ModernBertConfiguration.Activation

    init(_ c: ModernBertConfiguration) {
        activation = c.hiddenActivation
        _wi.wrappedValue = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: c.mlpBias)
        _wo.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: c.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = wi(x).split(parts: 2, axis: -1)
        let activated =
            switch activation {
            case .gelu: gelu(parts[0])
            case .silu: silu(parts[0])
            }
        return wo(activated * parts[1])
    }
}

// MARK: - Layer

/// Pre-norm encoder layer. Layer 0 has no attention norm: its input is the normalized embedding.
private final class ModernBertLayer: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm?
    @ModuleInfo(key: "attn") var attn: ModernBertAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: ModernBertMLP

    let isGlobal: Bool

    init(_ c: ModernBertConfiguration, index: Int) {
        isGlobal = c.globalLayers[index]
        if index > 0 {
            _attnNorm.wrappedValue = LayerNorm(
                dimensions: c.hiddenSize, eps: c.normEps, bias: c.normBias)
        }
        _attn.wrappedValue = ModernBertAttention(c, isGlobal: isGlobal)
        _mlpNorm.wrappedValue = LayerNorm(
            dimensions: c.hiddenSize, eps: c.normEps, bias: c.normBias)
        _mlp.wrappedValue = ModernBertMLP(c)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let h = x + attn(attnNorm.map { $0(x) } ?? x, mask: mask)
        return h + mlp(mlpNorm(h))
    }
}

// MARK: - Model

/// A ModernBERT encoder for text embeddings.
///
/// Returns the final hidden states; ``Pooling`` turns them into a sentence embedding, with CLS
/// pooling when the checkpoint has no `1_Pooling/config.json`. `maxPositionEmbeddings` stays
/// `nil`: inputs longer than the checkpoint's limit are the caller's to truncate.
public final class ModernBertModel: Module, EmbeddingModel {
    @ModuleInfo(key: "embeddings") private var embeddings: ModernBertEmbeddings
    @ModuleInfo(key: "layers") private var layers: [ModernBertLayer]
    @ModuleInfo(key: "final_norm") private var finalNorm: LayerNorm

    /// The model configuration.
    public let configuration: ModernBertConfiguration

    /// The size of the vocabulary.
    public var vocabularySize: Int { configuration.vocabularySize }

    /// CLS pooling, used when the checkpoint has no `1_Pooling/config.json`.
    public var poolingStrategy: Pooling.Strategy? { .cls }

    /// Creates the model.
    ///
    /// - Parameter configuration: The model configuration.
    public init(_ configuration: ModernBertConfiguration) {
        self.configuration = configuration
        _embeddings.wrappedValue = ModernBertEmbeddings(configuration)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            ModernBertLayer(configuration, index: $0)
        }
        _finalNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize, eps: configuration.normEps,
            bias: configuration.normBias)
        super.init()
    }

    /// Encodes the inputs.
    ///
    /// - Parameters:
    ///   - inputs: Token ids of shape `[Batch, Length]` or `[Length]`.
    ///   - positionIds: Ignored: RoPE positions start at 0.
    ///   - tokenTypeIds: Ignored.
    ///   - attentionMask: Optional padding mask with the shape of `inputs`, nonzero for real
    ///     tokens. Padded keys are hidden on every layer; padded batches require it.
    /// - Returns: The final hidden states `[Batch, Length, Hidden]` and no pooled output.
    public func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        EmbeddingModelOutput(
            hiddenStates: encode(inputs, attentionMask: attentionMask), pooledOutput: nil)
    }

    /// The embedding output, the output of every layer and the final hidden states, in order.
    /// For parity checks against the reference implementation.
    func layerStates(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> [MLXArray] {
        var states = [MLXArray]()
        let final = encode(inputs, attentionMask: attentionMask) { states.append($0) }
        return states + [final]
    }

    private func encode(
        _ inputs: MLXArray, attentionMask: MLXArray?, onState: (MLXArray) -> Void = { _ in }
    ) -> MLXArray {
        let ids = inputs.ndim == 1 ? inputs.reshaped(1, -1) : inputs
        // A 1-D mask goes with a 1-D input.
        let mask = attentionMask.map { $0.ndim == 1 ? $0.reshaped(1, -1) : $0 }
        if let mask {
            precondition(
                mask.shape == ids.shape,
                "attentionMask shape \(mask.shape) must match inputs \(ids.shape)")
        }

        var h = embeddings(ids)
        onState(h)
        let length = ids.dim(1)
        let globalMask = createBidirectionalAttentionMask(
            length: length, halfWindow: nil, paddingMask: mask)
        let localMask = createBidirectionalAttentionMask(
            length: length, halfWindow: configuration.localAttention / 2, paddingMask: mask)
        for layer in layers {
            h = layer(h, mask: layer.isGlobal ? globalMask : localMask)
            onState(h)
        }
        return finalNorm(h)
    }

    /// Accepts checkpoints saved as `ModernBertModel` (keys without prefix) and as
    /// `ModernBertForMaskedLM` (keys under `model.`): removes the prefix and drops the
    /// masked-LM head (`head.*`, `decoder.*`).
    ///
    /// - Parameter weights: The checkpoint weights.
    /// - Returns: The weights keyed as this model expects them.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("head.") || key.hasPrefix("decoder.") {
                continue
            }
            sanitized[key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key] = value
        }
        return sanitized
    }
}
