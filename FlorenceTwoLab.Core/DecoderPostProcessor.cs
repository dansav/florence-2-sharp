using System.Text.RegularExpressions;

using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Core;

// TODO: should split this class into multiple classes, one per task configuration
public partial class DecoderPostProcessor
{
    [GeneratedRegex(@"(\w+)(<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)+", RegexOptions.Compiled)]
    private static partial Regex CategoryAndRegionRegex();

    [GeneratedRegex(
        @"([^<]+)(?:<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)")]
    private static partial Regex CategoryAndQuadBoxRegex();

    [GeneratedRegex(@"<loc_(\d+)>")]
    private static partial Regex PointRegex();

    [GeneratedRegex(@"(\w+)<poly>(<loc_(\d+)>)+</poly>")]
    private static partial Regex LabeledPolygonsRegex();

    
    public async Task<Florence2Result> ProcessAsync(string modelOutput, Florence2TaskType taskType, bool imageWasPadded, int imageWidth, int imageHeight)
    {
        Florence2Tasks.TaskConfigurations.TryGetValue(taskType, out var taskConfig);

        // Florence2TaskType.OpenVocabularyDetection: arms<poly><loc_550><loc_421><loc_686><loc_510><loc_671><loc_740><loc_540><loc_616></poly>

        return taskConfig switch
        {
            // Advanced detection tasks, returns quad boxes.
            { ReturnsLabels: true, ReturnsBoundingBoxes: true, ReturnsPolygons: true } => await ProcessPointsAsQuadBoxesAsync(taskType, modelOutput, imageWasPadded, imageWidth, imageHeight),

            // Detection tasks
            { ReturnsLabels: true, ReturnsBoundingBoxes: true } => await ProcessPointsAsBoundingBoxesAsync(taskType, modelOutput, imageWasPadded, imageWidth, imageHeight),

            // Complex tasks
            { ReturnsLabels: true, ReturnsPolygons: true } => await ProcessLabeledPolygonsAsync(taskType, modelOutput, imageWasPadded, imageWidth, imageHeight),

            // Complex tasks
            { ReturnsPolygons: true } => await ProcessPointsAsPolygonsAsync(taskType, modelOutput, imageWasPadded, imageWidth, imageHeight),

            // Text generation tasks (captions, OCR)
            { ReturnsText: true } => new Florence2Result { TaskType = taskType, Text = modelOutput },

            // Region tasks - returns text probably
            // Florence2TaskType.RegionToDescription or
            //     Florence2TaskType.RegionToCategory or
            //     Florence2TaskType.RegionToOcr => await ProcessRegionResultAsync(taskType, modelOutput),

            _ => throw new ArgumentException($"Unsupported task type: {taskType}")
        };
    }

    private async Task<Florence2Result> ProcessPointsAsBoundingBoxesAsync(Florence2TaskType taskType, string modelOutput, bool imageWasPadded, int imageWidth, int imageHeight)
    {
        // NOTE: "wheel" has two bounding boxes, "door" has one
        // example data: car<loc_54><loc_375><loc_906><loc_707>door<loc_710><loc_276><loc_908><loc_537>wheel<loc_708><loc_557><loc_865><loc_704><loc_147><loc_563><loc_305><loc_705>
        // regex that parses one or more "(category)(one or more groups of 4 loc-tokens)"
        var regex = CategoryAndRegionRegex();

        var w = imageWidth / 1000f;
        var h = imageHeight / 1000f;

        List<string> labels = new();
        List<Rectangle> boundingBoxes = new();

        Match match = regex.Match(modelOutput);
        while (match.Success)
        {
            var label = match.Groups[1].Value;
            var captureCount = match.Groups[2].Captures.Count;
            for (int i = 0; i < captureCount; i++)
            {
                var x1 = int.Parse(match.Groups[3].Captures[i].Value);
                var y1 = int.Parse(match.Groups[4].Captures[i].Value);
                var x2 = int.Parse(match.Groups[5].Captures[i].Value);
                var y2 = int.Parse(match.Groups[6].Captures[i].Value);

                labels.Add(label);
                boundingBoxes.Add(new Rectangle(
                    (int)((0.5f + x1) * w),
                    (int)((0.5f + y1) * h),
                    (int)((x2 - x1) * w),
                    (int)((y2 - y1) * h)));
            }

            match = match.NextMatch();
        }

        return new Florence2Result { TaskType = taskType, BoundingBoxes = boundingBoxes, Labels = labels };
    }

    private async Task<Florence2Result> ProcessPointsAsQuadBoxesAsync(Florence2TaskType taskType, string modelOutput, bool imageWasPadded, int imageWidth, int imageHeight)
    {
        // Regex to match text followed by 8 location coordinates
        var regex = CategoryAndQuadBoxRegex();

        var matches = regex.Matches(modelOutput);

        var quadBoxes = new List<IReadOnlyCollection<Point>>();
        var labels = new List<string>();

        var w = imageWidth / 1000f;
        var h = imageHeight / 1000f;

        foreach (Match match in matches)
        {
            var text = match.Groups[1].Value;

            // Extract all 8 coordinates
            var points = new Point[4];
            for (int i = 0; i < 8; i += 2)
            {
                // Add 2 to group index because group[1] is the text
                var valueX = 0.5f + int.Parse(match.Groups[i + 2].Value);
                var valueY = 0.5f + int.Parse(match.Groups[i + 3].Value);

                // Convert from 0-1000 range to image coordinates
                points[i / 2] = new Point(
                    (int)(valueX * w),
                    (int)(valueY * h));
            }

            quadBoxes.Add(points);

            labels.Add(text);
        }

        // If you need to maintain compatibility with existing Rectangle format,
        // you could compute bounding rectangles that encompass each quad:
        var boundingBoxes = quadBoxes.Select(quad =>
        {
            var minX = quad.Min(p => p.X);
            var minY = quad.Min(p => p.Y);
            var maxX = quad.Max(p => p.X);
            var maxY = quad.Max(p => p.Y);

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

    private Task<Florence2Result> ProcessLabeledPolygonsAsync(Florence2TaskType taskType, string modelOutput, bool imageWasPadded, int imageWidth, int imageHeight)
    {
        var regex = LabeledPolygonsRegex();
        var match = regex.Match(modelOutput);
        
        var labels = new List<string>();
        var polygons = new List<IReadOnlyCollection<Point>>();
        var w = imageWidth / 1000f;
        var h = imageHeight / 1000f;
        
        while (match.Success)
        {
            var label = match.Groups[1].Value;
            var polygon = new List<Point>();
            var coordinates = match.Groups[3];
            
            for (int i = 0; i < coordinates.Captures.Count; i += 2)
            {
                var x = (int)((0.5f + int.Parse(coordinates.Captures[i].Value)) * w);
                var y = (int)((0.5f + int.Parse(coordinates.Captures[i + 1].Value)) * h);
                polygon.Add(new Point(x, y));
            }
            
            labels.Add(label);
            polygons.Add(polygon);
            
            match = match.NextMatch();
        }
        
        return Task.FromResult(new Florence2Result { TaskType = taskType, Labels = labels, Polygons = polygons });
    }
    
    private Task<Florence2Result> ProcessPointsAsPolygonsAsync(Florence2TaskType taskType, string modelOutput, bool imageWasPadded, int imageWidth, int imageHeight)
    {
        var regex = PointRegex();
        var matches = regex.EnumerateMatches(modelOutput);

        // for now, we only support a single polygon
        var polygons = new List<IReadOnlyCollection<Point>>();
        var polygon = new List<Point>();
        polygons.Add(polygon);

        var w = imageWidth / 1000f;
        var h = imageHeight / 1000f;

        // With match "<loc_XX>" the X is at index 5, and has the length match.Length - 5 - 1
        const int offset = 5;
        const int lengthOffset = 6;

        int count = 0;
        int x = 0;
        foreach (var match in matches)
        {
            var matchOffset = match.Index + offset;
            var matchLength = match.Length - lengthOffset;
            if (count % 2 == 0)
            {
                x = (int)((0.5f + int.Parse(modelOutput.AsSpan(matchOffset, matchLength))) * w);
            }
            else
            {
                var y = (int)((0.5f + int.Parse(modelOutput.AsSpan(matchOffset, matchLength))) * h);
                polygon.Add(new Point(x, y));
            }

            count++;
        }

        return Task.FromResult(new Florence2Result { TaskType = taskType, Polygons = polygons });
    }
}
