using System.Diagnostics;
using System.Reflection.Emit;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Core;

public partial class Florence2Pipeline
{
    private readonly ImageProcessor _imageProcessor;
    private readonly BartTokenizer _tokenizer;
    private readonly ModelRunner _modelRunner;
    private readonly EncoderPreProcessor _encoderPreprocessor;
    private readonly DecoderPostProcessor _postProcessor;

    private Florence2Pipeline(
        ImageProcessor imageProcessor,
        BartTokenizer tokenizer,
        ModelRunner modelRunner,
        EncoderPreProcessor encoderPreProcessor,
        DecoderPostProcessor postProcessor)
    {
        _imageProcessor = imageProcessor;
        _tokenizer = tokenizer;
        _modelRunner = modelRunner;
        _encoderPreprocessor = encoderPreProcessor;
        _postProcessor = postProcessor;
    }

    public static async Task<Florence2Pipeline> CreateAsync(Florence2Config config)
    {
        var imageProcessor = new ImageProcessor();
        var tokenizer = await BartTokenizer.FromPretrainedAsync(config.MetadataDirectory);
        var modelRunner = new ModelRunner(config);
        var encoderPreProcessor = new EncoderPreProcessor();
        var postProcessor = new DecoderPostProcessor();

        return new Florence2Pipeline(imageProcessor, tokenizer, modelRunner, encoderPreProcessor, postProcessor);
    }

    public async Task<Florence2Result> ProcessAsync(Image image, Florence2Query query)
    {
        var (taskType, prompt) = query;

        if (string.IsNullOrWhiteSpace(prompt))
        {
            throw new ArgumentException("Prompt cannot be empty");
        }

        // 1. Vision
        var processedImage = await _imageProcessor.ProcessImageAsync(image, false);
        var visionFeatures = await _modelRunner.RunVisionEncoderAsync(processedImage);

        // 2. Text
        var tokenized = _tokenizer.Tokenize(prompt);
        Debug.WriteLine($"Input tokens: '{string.Join("', '", tokenized)}'");

        var inputIds = new DenseTensor<long>(_tokenizer.ConvertTokensToIds(tokenized).Select(i => (long)i).ToArray(),
            [1, tokenized.Count]);
        var textFeatures = await _modelRunner.EmbedTokensAsync(inputIds);

        // 3. Concatenate vision and text features
        var (projectedFeatures, projectedAttentionMask) = _encoderPreprocessor.Process(visionFeatures, textFeatures, tokenized);

        // 4. Run encoder to get hidden states for decoder
        var encoderHiddenStates = await _modelRunner.RunEncoderAsync(projectedFeatures, projectedAttentionMask);

        // 5. Decoder in autoregressive mode to generate output text
        var decoderOutput = await _modelRunner.RunDecoderAsync(encoderHiddenStates, projectedAttentionMask);

        var text = _tokenizer.Decode(decoderOutput.Select(f => (int)f).ToList());

        // 6. Post-processing
        return await _postProcessor.ProcessAsync(text, taskType, true, image.Width, image.Height);
    }
}