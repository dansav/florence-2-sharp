using System.Diagnostics;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Core;

public class Florence2Pipeline
{
    private readonly Florence2ImageProcessor _imageProcessor;
    private readonly BartTokenizer _tokenizer;
    private readonly Florence2ModelRunner _modelRunner;

    private Florence2Pipeline(Florence2ImageProcessor imageProcessor, BartTokenizer tokenizer, Florence2ModelRunner modelRunner)
    {
        _imageProcessor = imageProcessor;
        _tokenizer = tokenizer;
        _modelRunner = modelRunner;
    }

    public static async Task<Florence2Pipeline> CreateAsync(Florence2Config config)
    {
        var imageProcessor = new Florence2ImageProcessor(config);
        var tokenizer = await BartTokenizer.FromPretrainedAsync(config.MetadataDirectory);
        var modelRunner = new Florence2ModelRunner(config);
        
        return new Florence2Pipeline(imageProcessor, tokenizer, modelRunner);
    }

    public async Task<Florence2Result> ProcessAsync(Image image, Florence2Query query)
    {
        var (taskType, prompt) = query;

        // 1. Vision Path
        var processedImage = await _imageProcessor.ProcessImageAsync(image);
        var visionFeatures = await _modelRunner.RunVisionEncoderAsync(processedImage);
        var visionAttentionMask = new DenseTensor<long>(
            Enumerable.Range(0, visionFeatures.Dimensions[1]).Select(f => 1L).ToArray(),
            [1, visionFeatures.Dimensions[1]]);
        Debug.Assert(visionFeatures.Dimensions[1] == visionAttentionMask.Dimensions[1]);

        // 2. Text Path
        var tokenized = _tokenizer.Tokenize(prompt);
        Console.WriteLine($"Input tokens: '{string.Join("', '", tokenized)}'");

        int[] shape = [ 1, tokenized.Count ];
        var textAttentionMask = new DenseTensor<long>(tokenized.Select(t => t == BartTokenizer.PadToken ? 0L : 1L).ToArray(), shape);
        var inputIds = new DenseTensor<long>(_tokenizer.ConvertTokensToIds(tokenized).Select(i => (long)i).ToArray(), shape);

        var textFeatures = await _modelRunner.EmbedTokensAsync(inputIds);
        Debug.Assert(textFeatures.Dimensions[1] == textAttentionMask.Dimensions[1]);

        // 3. Concatenate vision and text features
        var projectedFeatures = ConcatenateTensors(visionFeatures, textFeatures, 1);
        var projectedAttentionMask = ConcatenateTensors(visionAttentionMask, textAttentionMask, 1);

        // 4. Run encoder to get hidden states for decoder
        var encoderHiddenStates = await _modelRunner.RunEncoderAsync(projectedFeatures, projectedAttentionMask);

        // 5. Decoder in autoregressive mode to generate output text
        var decoderOutput = await _modelRunner.RunDecoderAsync(encoderHiddenStates, projectedAttentionMask);

        var text = _tokenizer.Decode(decoderOutput.Select(f => (int)f).ToList());

        // 6. Post-processing
        return await PostProcessResultAsync(text, taskType);
    }

    /// <summary>
    /// Concatenate two tensors along a specified axis
    /// </summary>
    /// <param name="tensor1">The first tensor to concatenate</param>
    /// <param name="tensor2">The second tensor to concatenate</param>
    /// <param name="axis">The axis along which to concatenate the tensors.</param>
    /// <typeparam name="T">The type of the tensor elements.</typeparam>
    /// <returns>
    /// The concatenated tensor.
    /// </returns>
    /// <exception cref="ArgumentException"></exception>
    private DenseTensor<T> ConcatenateTensors<T>(Tensor<T> tensor1, Tensor<T> tensor2, int axis)
    {
        if (tensor1.Rank != tensor2.Rank)
            throw new ArgumentException("Tensors must have the same number of dimensions");

        if (axis < 0 || axis >= tensor1.Rank)
            throw new ArgumentException("Invalid axis");

        if (axis != 1)
            throw new ArgumentException("Only concatenation along axis 1 is supported");

        var newDimensions = tensor1.Dimensions.ToArray();
        newDimensions[axis] += tensor2.Dimensions[axis];

        var result = new DenseTensor<T>(newDimensions);

        // Copy data from tensor1
        for (int i = 0; i < tensor1.Length; i++)
        {
            result.SetValue(i, tensor1.GetValue(i));
        }

        // Copy data from tensor2
        var offset = (int)tensor1.Length;
        for (int i = 0; i < tensor2.Length; i++)
        {
            result.SetValue(offset + i, tensor2.GetValue(i));
        }

        return result;
    }

    private async Task<Florence2Result> PostProcessResultAsync(string modelOutput, Florence2TaskType taskType)
    {
        return taskType switch
        {
            // Text generation tasks (captions, OCR)
            Florence2TaskType.Caption or
                Florence2TaskType.DetailedCaption or
                Florence2TaskType.MoreDetailedCaption or
                Florence2TaskType.Ocr => new Florence2Result { TaskType = taskType, Text = modelOutput },

            // Detection tasks
            Florence2TaskType.ObjectDetection or
                Florence2TaskType.DenseRegionCaption => await ProcessDetectionResultAsync(taskType, modelOutput),

            // Region tasks
            Florence2TaskType.RegionToDescription or
                Florence2TaskType.RegionToCategory or
                Florence2TaskType.RegionToOcr => await ProcessRegionResultAsync(taskType, modelOutput),

            // Complex tasks
            Florence2TaskType.ReferringExpressionSegmentation or
                Florence2TaskType.RegionToSegmentation => await ProcessSegmentationResultAsync(taskType, modelOutput),

            _ => throw new ArgumentException($"Unsupported task type: {taskType}")
        };
    }

    private Task<Florence2Result> ProcessDetectionResultAsync(Florence2TaskType taskType, string modelOutput)
    {
        // TODO: Implement detection result processing
        throw new NotImplementedException();
    }

    private Task<Florence2Result> ProcessRegionResultAsync(Florence2TaskType taskType, string modelOutput)
    {
        // TODO: Implement region result processing
        throw new NotImplementedException();
    }

    private Task<Florence2Result> ProcessSegmentationResultAsync(Florence2TaskType taskType, string modelOutput)
    {
        // TODO: Implement segmentation result processing
        throw new NotImplementedException();
    }
}