using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceTwoLab.Core;

internal sealed class ModelRunner : IDisposable
{
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _embedTokens;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _visionEncoder;

    public ModelRunner(IOnnxModelPathProvider pathProvider)
    {
        var modelDirectory = pathProvider.OnnxModelDirectory;

        // Create separate inference sessions for each model component
        var sessionOptions = new SessionOptions
        {
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        // Initialize sessions with model path
        _decoder = new InferenceSession(Path.Combine(modelDirectory, "decoder_model.onnx"), sessionOptions);
        _embedTokens = new InferenceSession(Path.Combine(modelDirectory, "embed_tokens.onnx"), sessionOptions);
        _encoder = new InferenceSession(Path.Combine(modelDirectory, "encoder_model.onnx"), sessionOptions);
        _visionEncoder = new InferenceSession(Path.Combine(modelDirectory, "vision_encoder.onnx"), sessionOptions);
    }

    public void Dispose()
    {
        _decoder.Dispose();
        _embedTokens.Dispose();
        _encoder.Dispose();
        _visionEncoder.Dispose();
    }

    public async Task<Tensor<float>> RunVisionEncoderAsync(DenseTensor<float> imageInput)
    {
        // Run vision encoder to get image features
        // Input: pixel_values [batch_size, 3, height, width]
        // Output: image_features [batch_size, sequence_length, 768]
        var visionInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", imageInput)
        };

        using var visionOutput = await RunInferenceAsync(_visionEncoder, visionInputs);
        var imageFeaturesTensor = visionOutput.First(o => o.Name == "image_features");
        return imageFeaturesTensor.Value as Tensor<float> ?? throw new InvalidCastException("image_features tensor is not of type Tensor<float>");
    }

    public async Task<Tensor<float>> EmbedTokensAsync(Tensor<long> tokens)
    {
        // Run token embedding model to get text features
        // Input: input_ids [batch_size, sequence_length]
        // Output: inputs_embeds [batch_size, sequence_length, 768]
        var embedInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", tokens)
        };

        using var embedOutput = await RunInferenceAsync(_embedTokens, embedInputs);
        var textFeaturesTensor = embedOutput.First(o => o.Name == "inputs_embeds").AsTensor<float>();
        return textFeaturesTensor;
    }

    public async Task<Tensor<float>> RunEncoderAsync(Tensor<float> embeddings, Tensor<long> attentionMask)
    {
        // Step 2: Run encoder on image features
        // Inputs: 
        // - inputs_embeds [batch_size, encoder_sequence_length, 768]
        // - attention_mask [batch_size, encoder_sequence_length]
        var encoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", embeddings),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask                )
        };

        using var encoderOutput = await RunInferenceAsync(_encoder, encoderInputs);
        var encoderHiddenStates = encoderOutput.First(o => o.Name == "last_hidden_state").AsTensor<float>();
        return encoderHiddenStates;
    }

    public async Task<IReadOnlyCollection<long>> RunDecoderAsync(Tensor<float> encoderHiddenStates, Tensor<long> encoderAttentionMask, int maxLength = 1024)
    {
        // this value comes from the "config.json" of the "onnx-community/Florence-2-*" repo.
        const int decoderStartTokenId = 2; // Initialize with decoder start token (end token?)
        const int eosTokenId = 2; // End of sentence token, TODO: we could get this from the tokenizer

        // Initialize with decoder start token
        var generatedTokens = new List<long> { decoderStartTokenId };

        // dry run???
        {
            // Create decoder inputs from current tokens
            var decoderInputIds = new DenseTensor<long>(
                generatedTokens.ToArray(),
                [1, generatedTokens.Count]
            );

            var decoderEmbeddings = await EmbedTokensAsync(decoderInputIds);

            // Run decoder
            NamedOnnxValue[] decoderInputs =
            [
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("encoder_attention_mask", encoderAttentionMask),
                NamedOnnxValue.CreateFromTensor("inputs_embeds", decoderEmbeddings)
            ];

            _ = await RunInferenceAsync(_decoder, decoderInputs);
            // var logits = outputs.First(o => o.Name == "logits").AsTensor<float>();
        }

        for (var i = 0; i < maxLength; i++)
        {
            // Create decoder inputs from current tokens
            var decoderInputIds = new DenseTensor<long>(
                generatedTokens.ToArray(),
                [1, generatedTokens.Count]
            );

            var decoderEmbeddings = await EmbedTokensAsync(decoderInputIds);

            // Run decoder
            NamedOnnxValue[] decoderInputs =
            [
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("encoder_attention_mask", encoderAttentionMask),
                NamedOnnxValue.CreateFromTensor("inputs_embeds", decoderEmbeddings)
            ];

            var outputs = await RunInferenceAsync(_decoder, decoderInputs);
            var logits = outputs.First(o => o.Name == "logits").AsTensor<float>();

            // Get next token (greedy selection from last position)
            var nextToken = GetNextToken(logits);

            // Stop if we hit EOS token
            if (nextToken == eosTokenId)
                break;

            generatedTokens.Add(nextToken);
        }

        return generatedTokens;
    }

    private static long GetNextToken(Tensor<float> logits)
    {
        // Get last position logits
        var lastLogits = logits.Dimensions[1] - 1;
        var vocabSize = logits.Dimensions[2];

        // Find max probability token
        var maxProb = float.MinValue;
        var maxToken = 0L;

        for (int i = 0; i < vocabSize; i++)
        {
            var prob = logits[0, lastLogits, i];
            if (prob > maxProb)
            {
                maxProb = prob;
                maxToken = i;
            }
        }

        return maxToken;
    }

    private static async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(
        InferenceSession session,
        IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        return await Task.Run(() => session.Run(inputs));
    }
}