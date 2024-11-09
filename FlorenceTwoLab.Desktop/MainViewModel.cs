using System;
using System.Collections.Generic;
using System.IO;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using CommunityToolkit.Mvvm.ComponentModel;
using FlorenceTwoLab.Core;
using FlorenceTwoLab.Core.Utils;
using System.Linq;
using System.Threading.Tasks;

namespace FlorenceTwoLab.Desktop;

public partial class MainViewModel : ObservableObject
{
    [ObservableProperty]
    private IEnumerable<string> _predefinedTasks = new[] { "Custom" }.Concat(Enum.GetNames(typeof(Florence2TaskType)));

    [ObservableProperty]
    private string _selectedTask;

    [ObservableProperty]
    private string? _customPrompt;

    [ObservableProperty]
    private string? _output;

    [ObservableProperty]
    private SixLabors.ImageSharp.Image? _preview;

    public MainViewModel()
    {
        SelectedTask = "Custom";
    }
    
    partial void OnSelectedTaskChanged(string value)
    {
        if (Preview is null) return;
        RunAsync();
    }

    public async Task LoadImageAsync(Stream stream)
    {
        var image = await SixLabors.ImageSharp.Image.LoadAsync(stream);
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

        if (!Enum.TryParse<Florence2TaskType>(SelectedTask, out var selectedTask))
        {
            // TODO: handle custom task type
            Console.WriteLine("Invalid task type");
        }
        
        var query = Florence2Tasks.CreateQuery(selectedTask);

        if (Preview is null)
        {
            Console.WriteLine("No image loaded");
            return;
        }
        
        try
        {
            var result = await pipeline.ProcessAsync(Preview, query);
            Console.WriteLine(result);
            
            Output = result.ToString();
        }
        catch (Exception exception)
        {
            Console.WriteLine(exception);
            if (Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                desktop.Shutdown();
            }
        }
    }
}