using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Florence2.Net;

internal sealed class Florence2ModelRunner : IDisposable
{
    private readonly InferenceSession _decoder;
    private readonly InferenceSession _embedTokens;
    private readonly InferenceSession _encoder;
    private readonly InferenceSession _visionEncoder;

    public Florence2ModelRunner(Florence2Config config)
    {
        var modelDirectory = config.OnnxModelDirectory;

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

    public async Task<Tensor<float>> RunDecoderAsync(Tensor<float> encoderHiddenStates, Tensor<long> encoderAttentionMask)
    {
        // Step 4: Run decoder
        // Inputs:
        // - encoder_hidden_states [batch_size, encoder_sequence_length, 768]
        // - encoder_attention_mask [batch_size, encoder_sequence_length]
        // - inputs_embeds [batch_size, decoder_sequence_length, 768]
        var decoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
            NamedOnnxValue.CreateFromTensor("encoder_attention_mask", encoderAttentionMask),
            NamedOnnxValue.CreateFromTensor("inputs_embeds", decoderInputEmbeddings)
        };

        var decoderOutput = await RunInferenceAsync(_decoder, decoderInputs);
        var logits = decoderOutput.First(o => o.Name == "logits").AsTensor<float>();
    }

    private static async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunInferenceAsync(
        InferenceSession session,
        IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        return await Task.Run(() => session.Run(inputs));
    }

    private static Tensor<long> CreateAttentionMask(int sequenceLength)
    {
        // Create attention mask tensor [batch_size, sequence_length]
        var tensor = new DenseTensor<long>(new[] { 1, sequenceLength });

        // Fill with 1s to attend to all tokens
        for (int i = 0; i < sequenceLength; i++)
        {
            tensor[0, i] = 1;
        }

        return tensor;
    }
}