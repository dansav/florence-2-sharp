using SixLabors.ImageSharp;

namespace Florence2.Net;

/// <summary>
/// Provides methods to create properly formatted prompts for all Florence-2 tasks
/// </summary>
public static class Florence2Tasks
{
    // Simple captioning tasks
    public static string CreateCaptionPrompt() => "<CAPTION>";
    public static string CreateDetailedCaptionPrompt() => "<DETAILED_CAPTION>";
    public static string CreateMoreDetailedCaptionPrompt() => "<MORE_DETAILED_CAPTION>";
    
    // OCR related tasks
    public static string CreateOcrPrompt() => "<OCR>";
    public static string CreateOcrWithRegionsPrompt() => "<OCR_WITH_REGION>";
    
    // Object detection tasks
    public static string CreateObjectDetectionPrompt() => "<OD>";
    public static string CreateDenseRegionCaptionPrompt() => "<DENSE_REGION_CAPTION>";
    public static string CreateRegionProposalPrompt() => "<REGION_PROPOSAL>";

    // Region analysis tasks
    public static string CreateRegionToDescriptionPrompt(Rectangle region, Size imageSize)
    {
        return $"<REGION_TO_DESCRIPTION>{region.CreateNormalizedRegionString(imageSize)}";
    }

    public static string CreateRegionToSegmentationPrompt(Rectangle region, Size imageSize)
    {
        return $"<REGION_TO_SEGMENTATION>{region.CreateNormalizedRegionString(imageSize)}";
    }

    public static string CreateRegionToCategoryPrompt(Rectangle region, Size imageSize)
    {
        return $"<REGION_TO_CATEGORY>{region.CreateNormalizedRegionString(imageSize)}";
    }

    public static string CreateRegionToOcrPrompt(Rectangle region, Size imageSize)
    {
        return $"<REGION_TO_OCR>{region.CreateNormalizedRegionString(imageSize)}";
    }

    // Grounding and referring expression tasks
    public static string CreateCaptionToGroundingPrompt(string caption)
    {
        return $"<CAPTION_TO_PHRASE_GROUNDING>{caption}";
    }

    public static string CreateReferringExpressionSegmentationPrompt(string expression)
    {
        return $"<REFERRING_EXPRESSION_SEGMENTATION>{expression}";
    }

    public static string CreateOpenVocabularyDetectionPrompt(string query)
    {
        return $"<OPEN_VOCABULARY_DETECTION>{query}";
    }

    // Cascaded tasks
    public static async Task<string> CreateCaptionWithGroundingPrompt(Florence2Pipeline pipeline, Image image)
    {
        // First get a caption
        var captionResult = await pipeline.ProcessAsync(image, CreateCaptionPrompt());
        var caption = captionResult.ToString();

        // Then create grounding prompt with that caption
        return CreateCaptionToGroundingPrompt(caption);
    }

    public static async Task<string> CreateDetailedCaptionWithGroundingPrompt(Florence2Pipeline pipeline, Image image)
    {
        var captionResult = await pipeline.ProcessAsync(image, CreateDetailedCaptionPrompt());
        var caption = captionResult.ToString();
        return CreateCaptionToGroundingPrompt(caption);
    }

    public static async Task<string> CreateMoreDetailedCaptionWithGroundingPrompt(Florence2Pipeline pipeline, Image image)
    {
        var captionResult = await pipeline.ProcessAsync(image, CreateMoreDetailedCaptionPrompt());
        var caption = captionResult.ToString();
        return CreateCaptionToGroundingPrompt(caption);
    }
}