using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

using FlorenceTwoLab.Core;
using FlorenceTwoLab.Core.Utils;
using FlorenceTwoLab.Desktop.Models;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

using Image = SixLabors.ImageSharp.Image;
using Point = SixLabors.ImageSharp.Point;
using Size = SixLabors.ImageSharp.Size;

namespace FlorenceTwoLab.Desktop.ViewModels;

public partial class MainViewModel : ObservableObject, IImageSelectionSource
{
    private readonly IReadOnlyCollection<ITaskGroupViewModel> _taskGroups;

    [ObservableProperty] private string? _output;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RunCommand), nameof(CreateRegionCommand))]
    private Image? _preview;

    [ObservableProperty] private bool _isPreviewVisible;

    [ObservableProperty] 
    [NotifyPropertyChangedFor(nameof(RegionSelectionEnabled))]
    [NotifyCanExecuteChangedFor(nameof(RunCommand), nameof(CreateRegionCommand), nameof(ClearRegionCommand))]
    private ITaskGroupViewModel? _selectedTaskGroup;

    private Florence2Pipeline? _pipeline;

    private Image? _loadedImage;

    public MainViewModel()
    {
        _taskGroups =
        [
            new BasicTaskGroupViewModel(),
            new RegionTaskGroupViewModel(this),
            new GroundingTaskGroupViewModel()
        ];

        ImageRegionSelector = new();
        ImageRegionSelector.PropertyChanged += (s, e) =>
        {
            if (e.PropertyName == nameof(ImageRegionSelector.HasRegions))
            {
                RunCommand.NotifyCanExecuteChanged();
                CreateRegionCommand.NotifyCanExecuteChanged();
                ClearRegionCommand.NotifyCanExecuteChanged();
                OnPropertyChanged(nameof(RegionSelectionEnabled));
            }
        };
        ImageRegionSelector.Regions.CollectionChanged += async (s, e) =>
        {
            if (e.NewItems is not null && e.NewItems.Count > 0)
            {
                var region = e.NewItems[0] as RegionOfInterest;
                if (region is null) return;
                ImageSelectionChanged?.Invoke(region, ImageSize);
                
                Preview = await DebugDecorateAsync(_loadedImage, region);
            }
        };
    }

    public event Action<RegionOfInterest, Size>? ImageSelectionChanged;

    public ImageRegionSelectorViewModel ImageRegionSelector { get; }

    public IEnumerable<ITaskGroupViewModel> TaskGroups => _taskGroups;

    public Size ImageSize => _loadedImage?.Size ?? new Size(0, 0);

    public bool RegionSelectionEnabled => SelectedTaskGroup is RegionTaskGroupViewModel;
    
    public async Task InitializeAsync()
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

        _pipeline = await Florence2Pipeline.CreateAsync(config);

        // string? testDataDir = Environment.GetEnvironmentVariable("FLORENCE2_TEST_DATA");
        // if (string.IsNullOrEmpty(testDataDir))
        // {
        //     var helper = new DataHelper();
        //     await helper.EnsureTestDataFilesAsync();
        //     testDataDir = helper.TestDataDirectory;
        // }

        // var testFile = Directory.GetFiles(testDataDir, "*.jpg")[0];
        // var image = SixLabors.ImageSharp.Image.Load(System.IO.Path.Combine(testDataDir, testFile));
    }

    partial void OnSelectedTaskGroupChanged(ITaskGroupViewModel? value)
    {
        value?.SelectFirstTask();
    }

    public async Task LoadImageAsync(Stream stream)
    {
        var image = await Image.LoadAsync(stream);
        _loadedImage = image;
        Preview = image;
    }
    
    [RelayCommand(CanExecute = nameof(CanRun))]
    private async Task RunAsync()
    {
        if (_pipeline is null)
        {
            Debug.WriteLine("Pipeline not initialized");
            return;
        }

        if (_loadedImage is null)
        {
            Debug.WriteLine("No image loaded");
            return;
        }

        Florence2Query? query = SelectedTaskGroup?.Query();
        if (string.IsNullOrWhiteSpace(query?.Prompt))
        {
            Debug.WriteLine("No prompt provided");
            return;
        }

        try
        {
            var result = await _pipeline.ProcessAsync(_loadedImage, query);
            Debug.WriteLine(result);

            if (result is { Polygons: not null, Labels: not null })
            {
                // text, bounding boxes, and polygons (only quad boxes are supported)
                Debug.WriteLine("Polygons and labels detected");
                Output = string.Join(", ", result.Labels);
                Preview = await DecorateAsync(_loadedImage, result.Labels, result.BoundingBoxes, result.Polygons);
                return;
            }

            if (result is { BoundingBoxes: not null, Labels: not null })
            {
                // text and bounding boxes
                Output = string.Join(", ", result.Labels!);
                Preview = await DecorateAsync(_loadedImage, result.Labels, result.BoundingBoxes);
                return;
            }

            if (result is { Polygons: not null })
            {
                // polygons (unknown if text)
                Output = $"{result.Polygons?.Count} polygons";
                Preview = await DecorateAsync(_loadedImage, polygons: result.Polygons);
                return;
            }

            if (result is { Text: not null })
            {
                // just text output
                Output = result.Text;
                Preview = _loadedImage;
                return;
            }

            Debug.WriteLine("Got result that could not be handled");
            throw new InvalidOperationException("Got result that could not be handled");
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
    
    [RelayCommand(CanExecute = nameof(CanCreateRegion))]
    private void CreateRegion(){}
    
    [RelayCommand(CanExecute = nameof(CanClearRegion))]
    private void ClearRegion()
    {
        ImageRegionSelector.ClearRegions();
    }
    
    private bool CanRun()
    {
        if (_pipeline is null)
        {
            Debug.WriteLine("Pipeline not initialized");
            return false;
        }
        
        if (_loadedImage is null)
        {
            Debug.WriteLine("No image loaded");
            return false;
        }

        return SelectedTaskGroup switch
        {
            BasicTaskGroupViewModel => true,
            GroundingTaskGroupViewModel grounding => grounding.HasCustomPrompt,
            RegionTaskGroupViewModel => ImageRegionSelector.HasRegions,
            _ => false
        };
    }

    private bool CanCreateRegion()
    {
        return Preview is not null && SelectedTaskGroup is RegionTaskGroupViewModel && !ImageRegionSelector.HasRegions;
    }
    
    private bool CanClearRegion()
    {
        return ImageRegionSelector.HasRegions;
    }
    
    private async Task<Image> DecorateAsync(
        Image image,
        List<string>? labels = null,
        List<Rectangle>? boundingBoxes = null,
        IReadOnlyCollection<IReadOnlyCollection<Point>>? polygons = null)
    {
        return await Task.Run(() =>
        {
            var image2 = image.CloneAs<Rgba32>();
            image2.Mutate(ctx =>
            {
                var chrome = new SolidBrush(Color.Red.WithAlpha(0.3f));
                var foreground = Color.White;

                if (boundingBoxes != null)
                {
                    for (int i = 0; i < boundingBoxes.Count; i++)
                    {
                        var rect = boundingBoxes[i];
                        ctx.DrawPolygon(chrome, 2f,
                            new PointF(rect.Left, rect.Top),
                            new PointF(rect.Right, rect.Top),
                            new PointF(rect.Right, rect.Bottom),
                            new PointF(rect.Left, rect.Bottom));

                        ctx.Fill(chrome, new RectangleF(rect.Left, rect.Bottom - 15, rect.Width, 15));

                        // only draw label if we have a bounding box
                        var label = labels?[i];
                        if (label != null)
                        {
                            ctx.DrawText(label, SystemFonts.CreateFont("Arial", 12), foreground,
                                new PointF(rect.Left + 5, rect.Bottom - 13));
                        }
                    }
                }

                if (polygons != null)
                {
                    var poly = new SolidBrush(Color.Yellow.WithAlpha(0.7f));
                    foreach (var polygon in polygons)
                    {
                        ctx.DrawPolygon(poly, 2f, polygon.Select(p => new PointF(p.X, p.Y)).ToArray());
                    }
                }
            });
            return image2;
        });
    }
    
    private async Task<Image?> DebugDecorateAsync(Image? loadedImage, RegionOfInterest region)
    {
        if (loadedImage is null) return null;

        var image = loadedImage.CloneAs<Rgba32>();
        
        image.Mutate(ctx =>
        {
            var w = image.Width;
            var h = image.Height;
            var chrome = new SolidBrush(Color.Green.WithAlpha(0.3f));
            var lineThickness = (float)Math.Sqrt(w * h) / 100f;
            var rect = region.RelativeBounds;
            
            ctx.DrawPolygon(chrome, lineThickness,
                new PointF(w * rect.Left, h * rect.Top),
                new PointF(w * rect.Right, h * rect.Top),
                new PointF(w * rect.Right, h * rect.Bottom),
                new PointF(w * rect.Left, h * rect.Bottom));
        });
        return image;
    }
}
