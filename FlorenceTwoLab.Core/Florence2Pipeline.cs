using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace Florence2.Net;

public class Florence2Pipeline
{
    private readonly Florence2Config _config;
    private readonly Florence2ImageProcessor _imageProcessor;
    private readonly BartTokenizer _tokenizer;
    private readonly Florence2ModelRunner _modelRunner;

    public Florence2Pipeline(Florence2Config config)
    {
        _config = config;
        _imageProcessor = new Florence2ImageProcessor(config);
        _tokenizer = new BartTokenizer(config);
        _modelRunner = new Florence2ModelRunner(config);
    }

    public async Task<Florence2Result> ProcessAsync(Image image, string prompt)
    {
        // 1. Vision Path
        var processedImage = await _imageProcessor.ProcessImageAsync(image);
        var visionFeatures = await _modelRunner.RunVisionEncoderAsync(processedImage);
        var visionAttentionMask = new DenseTensor<long>(visionFeatures.Select(f => 1L).ToArray(), new[] { 1, visionFeatures.Dimensions[1] });
        Debug.Assert(visionFeatures.Dimensions[1] == visionAttentionMask.Dimensions[1]);

        // 2. Text Path
        var (tokenized, textAttentionMask) = await _tokenizer.EncodeAsync(prompt);
        var textFeatures = await _modelRunner.EmbedTokensAsync(tokenized);
        Debug.Assert(textFeatures.Dimensions[1] == textAttentionMask.Dimensions[1]);

        // 3. Concatenate vision and text features
        var projectedFeatures = ConcatenateTensors(visionFeatures, textFeatures, 1);
        var projectedAttentionMask = ConcatenateTensors(visionAttentionMask, textAttentionMask, 1);
        
        // 4. Run encoder to get hidden states for decoder
        var encoderHiddenStates = await _modelRunner.RunEncoderAsync(projectedFeatures, projectedAttentionMask);

        // 5. Decoder in autoregressive mode to generate output text
        var decoderOutput = await _modelRunner.RunDecoderAsync(encoderHiddenStates, projectedAttentionMask);

        // 6. Post-processing
        var taskType = GetTaskType(prompt);
        return await PostProcessResultAsync(decoderOutput, taskType);
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
            result[i] = tensor1[i];
        }

        // Copy data from tensor2
        var offset = (int)tensor1.Length;
        for (int i = 0; i < tensor2.Length; i++)
        {
            result[offset + i] = tensor2[i];
        }

        return result;
    }

    private static Florence2TaskType GetTaskType(string prompt)
    {
        // Extract task type from prompt tags
        if (prompt.StartsWith("<CAPTION>")) return Florence2TaskType.Caption;
        if (prompt.StartsWith("<DETAILED_CAPTION>")) return Florence2TaskType.DetailedCaption;
        if (prompt.StartsWith("<MORE_DETAILED_CAPTION>")) return Florence2TaskType.MoreDetailedCaption;
        if (prompt.StartsWith("<OCR>")) return Florence2TaskType.Ocr;
        if (prompt.StartsWith("<OCR_WITH_REGION>")) return Florence2TaskType.OcrWithRegions;
        if (prompt.StartsWith("<OD>")) return Florence2TaskType.ObjectDetection;
        if (prompt.StartsWith("<DENSE_REGION_CAPTION>")) return Florence2TaskType.DenseRegionCaption;
        if (prompt.StartsWith("<REGION_PROPOSAL>")) return Florence2TaskType.RegionProposal;
        if (prompt.StartsWith("<CAPTION_TO_PHRASE_GROUNDING>")) return Florence2TaskType.CaptionToGrounding;
        if (prompt.StartsWith("<REFERRING_EXPRESSION_SEGMENTATION>"))
            return Florence2TaskType.ReferringExpressionSegmentation;
        if (prompt.StartsWith("<REGION_TO_SEGMENTATION>")) return Florence2TaskType.RegionToSegmentation;
        if (prompt.StartsWith("<OPEN_VOCABULARY_DETECTION>")) return Florence2TaskType.OpenVocabularyDetection;
        if (prompt.StartsWith("<REGION_TO_CATEGORY>")) return Florence2TaskType.RegionToCategory;
        if (prompt.StartsWith("<REGION_TO_DESCRIPTION>")) return Florence2TaskType.RegionToDescription;

        throw new ArgumentException($"Unknown task type in prompt: {prompt}");
    }

    private async Task<Florence2Result> PostProcessResultAsync(ModelOutput modelOutput, Florence2TaskType taskType)
    {
        // Get logits from model output
        var logits = modelOutput.GetLogits();

        return taskType switch
        {
            // Text generation tasks (captions, OCR)
            Florence2TaskType.Caption or
                Florence2TaskType.DetailedCaption or
                Florence2TaskType.MoreDetailedCaption or
                Florence2TaskType.Ocr => await ProcessTextGenerationResultAsync(taskType, logits),

            // Detection tasks
            Florence2TaskType.ObjectDetection or
                Florence2TaskType.DenseRegionCaption => await ProcessDetectionResultAsync(taskType, logits),

            // Region tasks
            Florence2TaskType.RegionToDescription or
                Florence2TaskType.RegionToCategory or
                Florence2TaskType.RegionToOcr => await ProcessRegionResultAsync(taskType, logits),

            // Complex tasks
            Florence2TaskType.ReferringExpressionSegmentation or
                Florence2TaskType.RegionToSegmentation => await ProcessSegmentationResultAsync(taskType, logits),

            _ => throw new ArgumentException($"Unsupported task type: {taskType}")
        };
    }

    private async Task<Florence2Result> ProcessTextGenerationResultAsync(Florence2TaskType taskType,
        Tensor<float> logits)
    {
        // For text generation tasks:
        // 1. Get highest probability tokens
        // 2. Convert tokens to text
        var result = new List<string>();

        await Task.Run(() =>
        {
            // Process each position in the sequence
            for (int pos = 0; pos < logits.Dimensions[1]; pos++)
            {
                // Get highest probability token for this position
                float maxProb = float.MinValue;
                int maxToken = 0;

                for (int token = 0; token < logits.Dimensions[2]; token++)
                {
                    var prob = logits[0, pos, token];
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxToken = token;
                    }
                }

                // Convert token to text
                // We need to implement token-to-text conversion in the tokenizer
                result.Add(_tokenizer.DecodeSingleToken(maxToken));
            }
        });

        return new Florence2Result
        {
            TaskType = taskType,
            Text = string.Join(" ", result).Trim()
        };
    }

    private Task<Florence2Result> ProcessDetectionResultAsync(Florence2TaskType taskType, Tensor<float> logits)
    {
        // TODO: Implement detection result processing
        throw new NotImplementedException();
    }

    private Task<Florence2Result> ProcessRegionResultAsync(Florence2TaskType taskType, Tensor<float> logits)
    {
        // TODO: Implement region result processing
        throw new NotImplementedException();
    }

    private Task<Florence2Result> ProcessSegmentationResultAsync(Florence2TaskType taskType, Tensor<float> logits)
    {
        // TODO: Implement segmentation result processing
        throw new NotImplementedException();
    }
}