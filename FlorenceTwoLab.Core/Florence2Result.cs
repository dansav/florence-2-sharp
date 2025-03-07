﻿using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Core;

public sealed class Florence2Result
{
    public Florence2TaskType TaskType { get; set; }

    // Text output (captions, OCR)
    public string? Text { get; set; }

    // Detection/region outputs
    public List<Rectangle>? BoundingBoxes { get; set; }
    public List<string>? Labels { get; set; }

    // Segmentation output
    public IReadOnlyCollection<IReadOnlyCollection<Point>>? Polygons { get; set; }

    public override string ToString()
    {
        var value = TaskType switch
        {
            Florence2TaskType.Caption => Text ?? string.Empty,
            Florence2TaskType.DetailedCaption => Text ?? string.Empty,
            Florence2TaskType.MoreDetailedCaption => Text ?? string.Empty,
            Florence2TaskType.Ocr => Text ?? string.Empty,
            Florence2TaskType.OcrWithRegions => Text ?? string.Empty,
            Florence2TaskType.ObjectDetection => string.Join(", ", Labels ?? Enumerable.Empty<string>()),
            Florence2TaskType.DenseRegionCaption => string.Join(", ", Labels ?? Enumerable.Empty<string>()),
            Florence2TaskType.RegionProposal => ZipLabelsAndBoundingBoxes(Labels, BoundingBoxes),
            Florence2TaskType.CaptionToGrounding => string.Join(", ", Labels ?? Enumerable.Empty<string>()),
            Florence2TaskType.ReferringExpressionSegmentation => string.Join(", ",
                (Polygons ?? Enumerable.Empty<IReadOnlyCollection<Point>>()).Select(p => string.Join(", ", p))),
            Florence2TaskType.RegionToSegmentation => string.Join(", ",
                (Polygons ?? Enumerable.Empty<IReadOnlyCollection<Point>>()).Select(p => string.Join(", ", p))),
            Florence2TaskType.OpenVocabularyDetection => string.Join(", ", Labels ?? Enumerable.Empty<string>()),
            Florence2TaskType.RegionToCategory => string.Join(", ", Labels ?? Enumerable.Empty<string>()),
            Florence2TaskType.RegionToDescription => Text ?? string.Empty,
            _ => string.Empty
        };

        return $"{TaskType}: {value}";
    }

    private static string ZipLabelsAndBoundingBoxes(IEnumerable<string>? label, IEnumerable<Rectangle>? boundingBox)
    {
        return string.Join(Environment.NewLine, (label ?? []).Zip((boundingBox ?? []), (l, b) => $"'{l}' -> [{b.Width}, {b.Height}]"));
    }
}