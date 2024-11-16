using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Core;

/// <summary>
/// Provides methods to create properly formatted prompts for all Florence-2 tasks
/// </summary>
public static class Florence2Tasks
{
    private static readonly Dictionary<Florence2TaskType, (string Token, string Prompt, bool RequiresRegion, bool RequiresSubPrompt)> TaskConfigurations = new()
    {
        // Basic captioning
        [Florence2TaskType.Caption] = ("<CAPTION>", "What does the image describe?", false, false),
        [Florence2TaskType.DetailedCaption] = ("<DETAILED_CAPTION>", "Describe in detail what is shown in the image.", false, false),
        [Florence2TaskType.MoreDetailedCaption] = ("<MORE_DETAILED_CAPTION>", "Describe with a paragraph what is shown in the image.", false, false),
        [Florence2TaskType.Ocr] = ("<OCR>", "What is the text in the image?", false, false),
        [Florence2TaskType.OcrWithRegions] = ("<OCR_WITH_REGION>", "What is the text in the image, with regions?", false, false),
        [Florence2TaskType.ObjectDetection] = ("<OD>", "Locate the objects with category name in the image.", false, false),
        [Florence2TaskType.DenseRegionCaption] = ("<DENSE_REGION_CAPTION>", "Locate the objects in the image, with their descriptions.", false, false),
        [Florence2TaskType.RegionProposal] = ("<REGION_PROPOSAL>", "Locate the region proposals in the image.", false, false),

        // Grounding and detection
        [Florence2TaskType.CaptionToGrounding] = ("<CAPTION_TO_PHRASE_GROUNDING>", "Locate the phrases in the caption: {0}", false, true),
        [Florence2TaskType.ReferringExpressionSegmentation] = ("<REFERRING_EXPRESSION_SEGMENTATION>", "Locate {0} in the image with mask", false, true),
        [Florence2TaskType.OpenVocabularyDetection] = ("<OPEN_VOCABULARY_DETECTION>", "Locate {0} in the image.", false, true),

        // Region analysis
        [Florence2TaskType.RegionToSegmentation] = ("<REGION_TO_SEGMENTATION>", "What is the polygon mask of region {0}", true, false),
        [Florence2TaskType.RegionToCategory] = ("<REGION_TO_CATEGORY>", "What is the region {0}?", true, false),
        [Florence2TaskType.RegionToDescription] = ("<REGION_TO_DESCRIPTION>", "What does the region {0} describe?", true, false),
        [Florence2TaskType.RegionToOcr] = ("<REGION_TO_OCR>", "What text is in the region {0}?", true, false)
    };

    private static readonly Dictionary<string, Florence2TaskType> TaskTypeLookup = TaskConfigurations.ToDictionary(x => x.Value.Token, x => x.Key);

    // public static Florence2Query CreateQuery(string customPrompt)
    // {
    //     return new Florence2Query(Florence2TaskType.Caption, customPrompt);
    // }

    /// <summary>
    /// Creates a query for the specified task type.
    /// </summary>
    /// <param name="taskType">
    /// The task type. Supported types are:
    /// - Caption
    /// - DetailedCaption
    /// - MoreDetailedCaption
    /// - Ocr
    /// - OcrWithRegions
    /// - ObjectDetection
    /// - DenseRegionCaption
    /// - RegionProposal
    /// </param>
    /// <returns>
    /// A query for the specified task type.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the task type is not supported.
    /// </exception>
    public static Florence2Query CreateQuery(Florence2TaskType taskType)
    {
        if (!TaskConfigurations.TryGetValue(taskType, out var config))
            throw new ArgumentException($"Unsupported task type: {taskType}");

        if (config.RequiresRegion)
            throw new ArgumentException($"Task {taskType} requires region parameter");

        if (config.RequiresSubPrompt)
            throw new ArgumentException($"Task {taskType} requires sub-prompt parameter");

        return new Florence2Query(taskType, config.Prompt);
    }

    /// <summary>
    /// Creates a query for the specified task type with the specified region. 
    /// </summary>
    /// <param name="taskType">
    /// The task type. Supported types are:
    /// - RegionToSegmentation
    /// - RegionToCategory
    /// - RegionToDescription
    /// - RegionToOcr 
    /// </param>
    /// <param name="region">
    /// The region of the image to query.
    /// </param>
    /// <param name="imageSize">
    /// The original size of the image.
    /// </param>
    /// <returns>
    /// A query for the specified task type with the specified region.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the task type is not supported.
    /// </exception>
    public static Florence2Query CreateQuery(Florence2TaskType taskType, Rectangle region, Size imageSize)
    {
        if (!TaskConfigurations.TryGetValue(taskType, out var config))
            throw new ArgumentException($"Unsupported task type: {taskType}");

        if (!config.RequiresRegion)
            throw new ArgumentException($"Task {taskType} does not handle region parameter");

        var regionString = region.CreateNormalizedRegionString(imageSize);
        return new Florence2Query(taskType, string.Format(config.Prompt, regionString));
    }

    /// <summary>
    /// Creates a query for the specified task type with the specified sub-prompt. 
    /// </summary>
    /// <param name="taskType">
    /// The task type. Supported types are:
    /// - CaptionToGrounding
    /// - ReferringExpressionSegmentation
    /// - OpenVocabularyDetection 
    /// </param>
    /// <param name="subPrompt">
    /// The sub-prompt to include in the query.
    /// </param>
    /// <returns>
    /// A query for the specified task type with the specified sub-prompt.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the task type is not supported.
    /// </exception>
    public static Florence2Query CreateQuery(Florence2TaskType taskType, string subPrompt)
    {
        if (!TaskConfigurations.TryGetValue(taskType, out var config))
            throw new ArgumentException($"Unsupported task type: {taskType}");

        if (!config.RequiresSubPrompt)
            throw new ArgumentException($"Task {taskType} does not handle input parameter");

        return new Florence2Query(taskType, string.Format(config.Prompt, subPrompt));
    }

    // Cascaded tasks
    public static async Task<Florence2Query> CreateQueryWithGroundingAsync(Florence2TaskType taskType, Florence2Pipeline pipeline, Image image)
    {
        switch (taskType)
        {
            case Florence2TaskType.Caption:
            case Florence2TaskType.DetailedCaption:
            case Florence2TaskType.MoreDetailedCaption:
                break;
            default:
                throw new ArgumentException($"Unsupported task type: {taskType}");
        }

        // First get a caption
        var initialQuery = CreateQuery(taskType);
        var captionResult = await pipeline.ProcessAsync(image, initialQuery);
        var caption = captionResult.ToString();

        // Then create grounding prompt with that caption
        return CreateQuery(Florence2TaskType.CaptionToGrounding, caption);
    }
}