using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using CommunityToolkit.Mvvm.ComponentModel;
using FlorenceTwoLab.Core;
using FlorenceTwoLab.Core.Utils;
using System.Linq;
using System.Threading.Tasks;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Point = SixLabors.ImageSharp.Point;

namespace FlorenceTwoLab.Desktop;

public partial class MainViewModel : ObservableObject
{
    private static readonly HashSet<string> _tasksWithCustomPrompt = new(["Custom"]);

    [ObservableProperty]
    private IEnumerable<string> _predefinedTasks = new[] { "Custom" }.Concat(Enum.GetNames(typeof(Florence2TaskType)));

    [ObservableProperty] [NotifyPropertyChangedFor(nameof(CanEditCustomPrompt))]
    private string _selectedTask;

    [ObservableProperty] private string? _customPrompt;

    [ObservableProperty] private string? _output;

    [ObservableProperty] private SixLabors.ImageSharp.Image? _preview;

    [ObservableProperty] private bool _isPreviewVisible;

    private Image? _loadedImage;

    public MainViewModel()
    {
        SelectedTask = Florence2TaskType.Caption.ToString();
    }

    public bool CanEditCustomPrompt => _tasksWithCustomPrompt.Contains(SelectedTask);

    async partial void OnSelectedTaskChanged(string value)
    {
        if (_loadedImage is null) return;
        try
        {
            await RunAsync();
        }
        catch (Exception e)
        {
            Debug.WriteLine(e);
            throw;
        }
    }

    public async Task LoadImageAsync(Stream stream)
    {
        var image = await SixLabors.ImageSharp.Image.LoadAsync(stream);
        _loadedImage = image;
        Preview = image;

        await RunAsync();
    }

    public async Task RunAsync()
    {
        string? modelDir = Environment.GetEnvironmentVariable("FLORENCE2_ONNX_MODELS");
        if (string.IsNullOrEmpty(modelDir))
        {
            var helper = new ModelHelper();
            await helper.EnsureModelFilesAsync();
            modelDir = helper.ModelDirectory;
        }

        var config = new Florence2Config
        {
            OnnxModelDirectory = modelDir ?? throw new NullReferenceException("A model directory is required"),
            MetadataDirectory = System.IO.Path.Combine(modelDir, ".."),
        };

        var pipeline = await Florence2Pipeline.CreateAsync(config);

        // string? testDataDir = Environment.GetEnvironmentVariable("FLORENCE2_TEST_DATA");
        // if (string.IsNullOrEmpty(testDataDir))
        // {
        //     var helper = new DataHelper();
        //     await helper.EnsureTestDataFilesAsync();
        //     testDataDir = helper.TestDataDirectory;
        // }

        // var testFile = Directory.GetFiles(testDataDir, "*.jpg")[0];
        // var image = SixLabors.ImageSharp.Image.Load(System.IO.Path.Combine(testDataDir, testFile));

        Florence2Query query;
        if (Enum.TryParse<Florence2TaskType>(SelectedTask, out var selectedTask))
        {
            query = Florence2Tasks.CreateQuery(selectedTask);
        }
        else
        {
            query = new Florence2Query(Florence2TaskType.Caption, CustomPrompt ?? "");
        }

        if (_loadedImage is null)
        {
            Debug.WriteLine("No image loaded");
            return;
        }

        if (string.IsNullOrWhiteSpace(query.Prompt))
        {
            Debug.WriteLine("No prompt provided");
            return;
        }

        try
        {
            var result = await pipeline.ProcessAsync(_loadedImage, query);
            Debug.WriteLine(result);

            switch (result.TaskType)
            {
                case Florence2TaskType.Caption:
                case Florence2TaskType.DetailedCaption:
                case Florence2TaskType.MoreDetailedCaption:
                case Florence2TaskType.Ocr:
                    Output = result.ToString();
                    Preview = _loadedImage;
                    break;
                case Florence2TaskType.OcrWithRegions:
                    Output = string.Join(", ", result.Labels!);
                    Preview = await DecorateAsync(_loadedImage, result.Labels!, result.BoundingBoxes!, result.Polygons!);
                    break;
                case Florence2TaskType.ObjectDetection:
                    Output = string.Join(", ", result.Labels!);
                    Preview = await DecorateAsync(_loadedImage, result.Labels!, result.BoundingBoxes!);
                    break;
                case Florence2TaskType.DenseRegionCaption:
                    break;
                case Florence2TaskType.RegionProposal:
                    break;
                case Florence2TaskType.RegionToDescription:
                    break;
                case Florence2TaskType.RegionToSegmentation:
                    break;
                case Florence2TaskType.RegionToCategory:
                    break;
                case Florence2TaskType.RegionToOcr:
                    break;
                case Florence2TaskType.CaptionToGrounding:
                    break;
                case Florence2TaskType.ReferringExpressionSegmentation:
                    break;
                case Florence2TaskType.OpenVocabularyDetection:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
        catch (Exception exception)
        {
            Debug.WriteLine(exception);
            if (Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                desktop.Shutdown();
            }
        }
    }

    private async Task<Image> DecorateAsync(Image image, List<string> labels, List<Rectangle> boundingBoxes)
    {
        return await Task.Run(() =>
        {
            var image2 = image.CloneAs<Rgba32>();
            image2.Mutate(ctx =>
            {
                var chrome = new SolidBrush(Color.Red.WithAlpha(0.3f));
                var foreground = Color.White;

                for (int i = 0; i < boundingBoxes.Count; i++)
                {
                    var rect = boundingBoxes[i];
                    var label = labels[i];
                    ctx.DrawPolygon(chrome, 2f,
                        new PointF(rect.Left, rect.Top),
                        new PointF(rect.Right, rect.Top),
                        new PointF(rect.Right, rect.Bottom),
                        new PointF(rect.Left, rect.Bottom));
                    
                    ctx.Fill( chrome, new RectangleF(rect.Left, rect.Bottom - 15, rect.Width, 15));
                    ctx.DrawText(label, SystemFonts.CreateFont("Arial", 12), foreground,
                        new PointF(rect.Left + 5, rect.Bottom - 13));
                }
            });
            return image2;
        });
    }
    
    private async Task<Image> DecorateAsync(Image image, List<string> labels, List<Rectangle> boundingBoxes, IReadOnlyCollection<IReadOnlyCollection<Point>> polygons)
    {
        return await Task.Run(() =>
        {
            var image2 = image.CloneAs<Rgba32>();
            image2.Mutate(ctx =>
            {
                var chrome = new SolidBrush(Color.Red.WithAlpha(0.3f));
                var poly = new SolidBrush(Color.Yellow.WithAlpha(0.7f));
                var foreground = Color.White;

                for (int i = 0; i < boundingBoxes.Count; i++)
                {
                    var rect = boundingBoxes[i];
                    var label = labels[i];
                    ctx.DrawPolygon(chrome, 2f,
                        new PointF(rect.Left, rect.Top),
                        new PointF(rect.Right, rect.Top),
                        new PointF(rect.Right, rect.Bottom),
                        new PointF(rect.Left, rect.Bottom));
                    
                    ctx.Fill( chrome, new RectangleF(rect.Left, rect.Bottom - 15, rect.Width, 15));
                    ctx.DrawText(label, SystemFonts.CreateFont("Arial", 12), foreground,
                        new PointF(rect.Left + 5, rect.Bottom - 13));
                }

                foreach (var polygon in polygons)
                {
                    ctx.DrawPolygon(poly, 2f, polygon.Select(p => new PointF(p.X, p.Y)).ToArray());
                }
            });
            return image2;
        });
    }
}