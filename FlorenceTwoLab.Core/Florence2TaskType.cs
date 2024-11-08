namespace FlorenceTwoLab.Core;

/// <summary>
/// Enumerates the supported Florence-2 task types
/// </summary>
public enum Florence2TaskType
{
    // Basic captioning
    Caption,
    DetailedCaption,
    MoreDetailedCaption,

    // OCR
    Ocr,
    OcrWithRegions,

    // Object detection
    ObjectDetection,
    DenseRegionCaption,
    RegionProposal,

    // Region analysis
    RegionToDescription,
    RegionToSegmentation,
    RegionToCategory,
    RegionToOcr,

    // Grounding and detection
    CaptionToGrounding,
    ReferringExpressionSegmentation,
    OpenVocabularyDetection
}