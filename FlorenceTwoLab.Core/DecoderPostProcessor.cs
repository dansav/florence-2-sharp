using System.Text.RegularExpressions;
using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Core;

public partial class DecoderPostProcessor
{
    [GeneratedRegex(@"(\w+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")]
    private static partial Regex CategoryAndRegionRegex();

    [GeneratedRegex(
        @"([^<]+)(?:<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)")]
    private static partial Regex CategoryAndQuadBoxRegex();

    public async Task<Florence2Result> ProcessAsync(string modelOutput, Florence2TaskType taskType, bool imageWasPadded,
        int imageWidth, int imageHeight)
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
                Florence2TaskType.DenseRegionCaption or
                Florence2TaskType.RegionProposal or
                Florence2TaskType.CaptionToGrounding => await ProcessDetectionResultAsync(taskType, modelOutput,
                    imageWasPadded, imageWidth, imageHeight),

            // Advanced detection tasks
            Florence2TaskType.OcrWithRegions or Florence2TaskType.OpenVocabularyDetection => await
                ProcessDetection2ResultAsync(taskType, modelOutput,
                    imageWasPadded, imageWidth, imageHeight),

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

    private async Task<Florence2Result> ProcessDetectionResultAsync(Florence2TaskType taskType, string modelOutput,
        bool imageWasPadded, int imageWidth, int imageHeight)
    {
        // TODO: Implement detection result processing
        // example data: car<loc_54><loc_375><loc_906><loc_707>door<loc_710><loc_276><loc_908><loc_537>wheel<loc_708><loc_557><loc_865><loc_704><loc_147><loc_563><loc_305><loc_705>

        // regex that parses one or more "(category)<loc_(x1)><loc_(y1)><loc_(x2)><loc_(y2)>"
        var regex = CategoryAndRegionRegex();
        (string Label, int X1, int Y1, int X2, int Y2)[] regions = regex.Matches(modelOutput)
            .Select(m => (
                m.Groups[1].Value,
                int.Parse(m.Groups[2].Value),
                int.Parse(m.Groups[3].Value),
                int.Parse(m.Groups[4].Value),
                int.Parse(m.Groups[5].Value)
            ))
            .ToArray();

        var labels = regions.Select(r => r.Label).ToList();

        var boundingBoxes = regions
            .Select(r => new Rectangle(
                (int)(r.X1 * 0.001f * imageWidth),
                (int)(r.Y1 * 0.001f * imageHeight),
                (int)((r.X2 - r.X1) * 0.001f * imageWidth),
                (int)((r.Y2 - r.Y1) * 0.001f * imageHeight)))
            .ToList();

        return new Florence2Result { TaskType = taskType, BoundingBoxes = boundingBoxes, Labels = labels };
    }

    private async Task<Florence2Result> ProcessDetection2ResultAsync(Florence2TaskType taskType, string modelOutput,
        bool imageWasPadded, int imageWidth, int imageHeight)
    {
        // Regex to match text followed by 8 location coordinates
        var regex = CategoryAndQuadBoxRegex();

        var matches = regex.Matches(modelOutput);

        var quadBoxes = new List<IReadOnlyCollection<Point>>();
        var labels = new List<string>();

        foreach (Match match in matches)
        {
            var text = match.Groups[1].Value;

            // Extract all 8 coordinates
            var points = new Point[4];
            for (int i = 0; i < 8; i += 2)
            {
                // Add 2 to group index because group[1] is the text
                var valueX = int.Parse(match.Groups[i + 2].Value);
                var valueY = int.Parse(match.Groups[i + 3].Value);

                // Convert from 0-1000 range to image coordinates
                points[i / 2] = new Point(
                    (int)(valueX * 0.001f * imageWidth),
                    (int)(valueY * 0.001f * imageHeight));
            }

            quadBoxes.Add(points);

            labels.Add(text);
        }

        // If you need to maintain compatibility with existing Rectangle format,
        // you could compute bounding rectangles that encompass each quad:
        var boundingBoxes = quadBoxes.Select(quad =>
        {
            var minX = (int)quad.Min(p => p.X);
            var minY = (int)quad.Min(p => p.Y);
            var maxX = (int)quad.Max(p => p.X);
            var maxY = (int)quad.Max(p => p.Y);

            return new Rectangle(minX, minY, maxX - minX, maxY - minY);
        }).ToList();

        return new Florence2Result
        {
            TaskType = taskType,
            BoundingBoxes = boundingBoxes,
            Labels = labels,
            Polygons = quadBoxes
        };
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