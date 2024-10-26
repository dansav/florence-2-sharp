using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace Florence2.Net;

public class Florence2Pipeline
{
    private readonly Florence2Config _config;
    private readonly Florence2ImageProcessor _imageProcessor;
    private readonly Florence2Tokenizer _tokenizer;
    private readonly Florence2ModelRunner _modelRunner;

    public Florence2Pipeline(Florence2Config config)
    {
        _config = config;
        _imageProcessor = new Florence2ImageProcessor(config);
        _tokenizer = new Florence2Tokenizer(config);
        _modelRunner = new Florence2ModelRunner(config);
    }

    public async Task<Florence2Result> ProcessAsync(Image image, string prompt)
    {
        // Get task type from prompt
        var taskType = GetTaskType(prompt);
        
        // Process image
        var processedImage = await _imageProcessor.ProcessImageAsync(image);
        
        // Tokenize text
        var tokenized = await _tokenizer.EncodeAsync(prompt);
        
        // Run model
        var modelOutput = await _modelRunner.RunAsync(processedImage, tokenized);
        
        // Post-process results
        return await PostProcessResultAsync(modelOutput, taskType);
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
        if (prompt.StartsWith("<REFERRING_EXPRESSION_SEGMENTATION>")) return Florence2TaskType.ReferringExpressionSegmentation;
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
    
    private async Task<Florence2Result> ProcessTextGenerationResultAsync(Florence2TaskType taskType, Tensor<float> logits)
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