namespace Florence2.Net;

/// <summary>
/// Extension methods for working with Florence2TaskType
/// </summary>
public static class Florence2TaskTypeExtensions
{
    public static bool RequiresRegion(this Florence2TaskType taskType)
    {
        return taskType switch
        {
            Florence2TaskType.RegionToDescription => true,
            Florence2TaskType.RegionToSegmentation => true,
            Florence2TaskType.RegionToCategory => true,
            Florence2TaskType.RegionToOcr => true,
            _ => false
        };
    }

    public static bool RequiresTextInput(this Florence2TaskType taskType)
    {
        return taskType switch
        {
            Florence2TaskType.CaptionToGrounding => true,
            Florence2TaskType.ReferringExpressionSegmentation => true,
            Florence2TaskType.OpenVocabularyDetection => true,
            _ => false
        };
    }

    public static bool IsCascaded(this Florence2TaskType taskType)
    {
        // Tasks that require running another task first
        return taskType switch
        {
            Florence2TaskType.CaptionToGrounding => true,
            _ => false
        };
    }
}