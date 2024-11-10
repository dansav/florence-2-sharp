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
using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Desktop;

public partial class MainViewModel : ObservableObject
{
    private static readonly HashSet<string> _tasksWithCustomPrompt = new(["Custom"]);
    
    [ObservableProperty]
    private IEnumerable<string> _predefinedTasks = new[] { "Custom" }.Concat(Enum.GetNames(typeof(Florence2TaskType)));

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanEditCustomPrompt))]
    private string _selectedTask;

    [ObservableProperty]
    private string? _customPrompt;
    
    [ObservableProperty]
    private string? _output;

    [ObservableProperty]
    private SixLabors.ImageSharp.Image? _preview;

    [ObservableProperty]
    private bool _isPreviewVisible;

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
            
            Output = result.ToString();
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
}