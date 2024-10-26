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
        _decoder = new InferenceSession(Path.Combine(modelDirectory, "decoder_model.onnx"), sessionOptions );
        _embedTokens = new InferenceSession(Path.Combine(modelDirectory, "embed_tokens.onnx"), sessionOptions );
        _encoder = new InferenceSession(Path.Combine(modelDirectory, "encoder_model.onnx"), sessionOptions );
        _visionEncoder = new InferenceSession(Path.Combine(modelDirectory, "vision_encoder.onnx"), sessionOptions );
    }
    
    public void Dispose()
    {
        _decoder.Dispose();
        _embedTokens.Dispose();
        _encoder.Dispose();
        _visionEncoder.Dispose();
    } 

    public async Task<ModelOutput> RunAsync(DenseTensor<float> imageFeatures, Tensor<long> inputIds)
    {
        // Step 1: Run vision encoder to get image features
        // Input: pixel_values [batch_size, 3, height, width]
        // Output: image_features [batch_size, sequence_length, 768]
        var visionInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", imageFeatures)
        };

        using var visionOutput = await RunInferenceAsync(_visionEncoder, visionInputs);
        var imageFeaturesTensor = visionOutput.First(o => o.Name == "image_features").AsTensor<float>();

        // Step 2: Run encoder on image features
        // Inputs: 
        // - inputs_embeds [batch_size, encoder_sequence_length, 768]
        // - attention_mask [batch_size, encoder_sequence_length]
        var encoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", imageFeaturesTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", 
                CreateAttentionMask(imageFeaturesTensor.Dimensions[1]))
        };

        using var encoderOutput = await RunInferenceAsync(_encoder, encoderInputs);
        var encoderHiddenStates = encoderOutput.First(o => o.Name == "last_hidden_state").AsTensor<float>();

        // Step 3: Embed decoder input tokens
        // Input: input_ids [batch_size, sequence_length]
        // Output: inputs_embeds [batch_size, sequence_length, 768]
        var embedInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds)
        };

        using var embedOutput = await RunInferenceAsync(_embedTokens, embedInputs);
        var decoderInputEmbeddings = embedOutput.First(o => o.Name == "inputs_embeds").AsTensor<float>();

        // Step 4: Run decoder
        // Inputs:
        // - encoder_hidden_states [batch_size, encoder_sequence_length, 768]
        // - encoder_attention_mask [batch_size, encoder_sequence_length]
        // - inputs_embeds [batch_size, decoder_sequence_length, 768]
        var decoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
            NamedOnnxValue.CreateFromTensor("encoder_attention_mask", 
                CreateAttentionMask(encoderHiddenStates.Dimensions[1])),
            NamedOnnxValue.CreateFromTensor("inputs_embeds", decoderInputEmbeddings)
        };

        var decoderOutput = await RunInferenceAsync(_decoder, decoderInputs);
        
        return new ModelOutput(decoderOutput);
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