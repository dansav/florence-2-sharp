using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Model;

public class Florence2
{
    private InferenceSession _visionEncoderSession;
    private InferenceSession _encoderSession;
    private InferenceSession _decoderSession;
    private InferenceSession _embedTokensSession;
    private AutoTokenizer _tokenizer;
    private AutoProcessor _imageProcessor;

    public Florence2()
    {
        _tokenizer = new AutoTokenizer();
        _imageProcessor = new AutoProcessor();
    }
    
    public async Task InitializeAsync()
    {
        var modelDir = "Model";
        _visionEncoderSession = new InferenceSession(Path.Combine(modelDir, "vision_encoder.onnx"));
        _encoderSession = new InferenceSession(Path.Combine(modelDir, "encoder_model.onnx"));
        _decoderSession = new InferenceSession(Path.Combine(modelDir, "decoder_model.onnx"));
        _embedTokensSession = new InferenceSession(Path.Combine(modelDir, "embed_tokens.onnx"));

        // Initialize tokenizer and image processor
        await _tokenizer.InitializeAsync();
        await _imageProcessor.InitializeAsync();
    }

    public async Task<string> GenerateImageDescriptionAsync(string imagePath)
    {
        // Load and preprocess the image
        using var image = await Image.LoadAsync(imagePath);
        var processedImage = _imageProcessor.PreprocessImage(image);

        // Run vision encoder
        var visionEncoderInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("pixel_values", processedImage) };
        var visionEncoderOutputs = _visionEncoderSession.Run(visionEncoderInputs);
        var imageFeatures = visionEncoderOutputs.First(x => x.Name == "image_features").AsTensor<float>();

        // Run encoder
        var encoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs_embeds", imageFeatures),
            NamedOnnxValue.CreateFromTensor("attention_mask", CreateAttentionMask(imageFeatures.Dimensions[1]))
        };
        var encoderOutputs = _encoderSession.Run(encoderInputs);
        var encoderHiddenStates = encoderOutputs.First(x => x.Name == "last_hidden_state").AsTensor<float>();

        // Initialize decoder input
        var decoderInput = _tokenizer.Encode("<start_of_text>");
        var generatedText = new List<int>();

        // Generate text token by token
        for (int i = 0; i < 50; i++) // Max length of 50 tokens
        {
            var inputEmbeddings = RunEmbedTokens(decoderInput);
            
            var decoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("encoder_attention_mask", CreateAttentionMask(encoderHiddenStates.Dimensions[1])),
                NamedOnnxValue.CreateFromTensor("inputs_embeds", inputEmbeddings)
            };

            var decoderOutputs = _decoderSession.Run(decoderInputs);
            var logits = decoderOutputs.First(x => x.Name == "logits").AsTensor<float>();

            var nextToken = GetNextToken(logits);
            if (nextToken == _tokenizer.EndToken) break;

            generatedText.Add(nextToken);
            decoderInput = new[] { nextToken };
        }

        return _tokenizer.Decode(generatedText);
    }

    private Tensor<float> RunEmbedTokens(int[] inputIds)
    {
        var embedInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", TensorFromArray(inputIds)) };
        var embedOutputs = _embedTokensSession.Run(embedInputs);
        return embedOutputs.First(x => x.Name == "inputs_embeds").AsTensor<float>();
    }

    private Tensor<long> CreateAttentionMask(int length)
    {
        return TensorFromArray(Enumerable.Repeat(1L, length).ToArray());
    }

    private int GetNextToken(Tensor<float> logits)
    {
        // Implement token selection logic (e.g., argmax or sampling)
        // For simplicity, we'll use argmax here
        return logits.AsEnumerable().Select((value, index) => new { Value = value, Index = index })
            .OrderByDescending(x => x.Value)
            .First().Index;
    }

    private Tensor<T> TensorFromArray<T>(T[] array) where T : struct
    {
        return new DenseTensor<T>(array, new[] { 1, array.Length });
    }
}

/*

decoder_model.onnx:
  Input Metadata:
    - Name: encoder_attention_mask
      Dimensions: System.Int64[batch_size, encoder_sequence_length]
    - Name: encoder_hidden_states
      Dimensions: System.Single[batch_size, encoder_sequence_length, ]
    - Name: inputs_embeds
      Dimensions: System.Single[batch_size, decoder_sequence_length, ]
  Output Metadata:
    - Name: logits
      Dimensions: System.Single[batch_size, decoder_sequence_length, ]
    - Name: present.0.decoder.key
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.0.decoder.value
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.0.encoder.key
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.0.encoder.value
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.1.decoder.key
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.1.decoder.value
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.1.encoder.key
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.1.encoder.value
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.2.decoder.key
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.2.decoder.value
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.2.encoder.key
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.2.encoder.value
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.3.decoder.key
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.3.decoder.value
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.3.encoder.key
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.3.encoder.value
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.4.decoder.key
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.4.decoder.value
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.4.encoder.key
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.4.encoder.value
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.5.decoder.key
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.5.decoder.value
      Dimensions: System.Single[batch_size, , decoder_sequence_length, ]
    - Name: present.5.encoder.key
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]
    - Name: present.5.encoder.value
      Dimensions: System.Single[batch_size, , encoder_sequence_length, ]

embed_tokens.onnx:
  Input Metadata:
    - Name: input_ids
      Dimensions: System.Int64[batch_size, sequence_length]
  Output Metadata:
    - Name: inputs_embeds
      Dimensions: System.Single[batch_size, sequence_length, ]

encoder_model.onnx:
  Input Metadata:
    - Name: attention_mask
      Dimensions: System.Int64[batch_size, encoder_sequence_length]
    - Name: inputs_embeds
      Dimensions: System.Single[batch_size, encoder_sequence_length, ]
  Output Metadata:
    - Name: last_hidden_state
      Dimensions: System.Single[batch_size, encoder_sequence_length, ]

vision_encoder.onnx:
  Input Metadata:
    - Name: pixel_values
      Dimensions: System.Single[batch_size, , height, width]
  Output Metadata:
    - Name: image_features
      Dimensions: System.Single[batch_size, floor(Reshape_2813_o0__d0*Reshape_2813_o0__d1*Reshape_2813_o0__d2/batch_size) + 1, ]



 */
