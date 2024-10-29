using System;
using System.IO;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Interactivity;
using Florence2.Net;

namespace FlorenceTwoLab.Desktop;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
    }

    private async void MainWindow_Loaded(object? sender, RoutedEventArgs e)
    {
        // foreach (var value in Enum.GetValues<Environment.SpecialFolder>())
        // {
        //     var folder = Environment.GetFolderPath(value);
        //     Console.WriteLine($"{value}: {folder}");
        // }
        // return;

        var modelDir = Environment.GetEnvironmentVariable("FLORENCE2_ONNX_MODELS");
        if (string.IsNullOrEmpty(modelDir))
        {
            var helper = new DataHelper();
            await helper.EnsureModelFilesAsync();
            modelDir = helper.ModelDirectory;
        }

        // var florence2 = new Florence2_Old();
        // florence2.Initialize(modelDir);

        var config = new Florence2Config
        {
            OnnxModelDirectory = modelDir,
            MetadataDirectory = System.IO.Path.Combine(modelDir, ".."),
        };
        
        var pipeline = new Florence2Pipeline(config);
        
        var testDataDir = Environment.GetEnvironmentVariable("FLORENCE2_TEST_DATA");
        if (string.IsNullOrEmpty(testDataDir))
        {
            var helper = new DataHelper();
            await helper.EnsureTestDataFilesAsync();
            testDataDir = helper.TestDataDirectory;
        }
        
        var testFile = Directory.GetFiles(testDataDir, "*.jpg")[0];
        
        var image = SixLabors.ImageSharp.Image.Load(System.IO.Path.Combine(testDataDir, testFile));

        var prompt = Florence2Tasks.CreateCaptionPrompt();

        try
        {
            var result = await pipeline.ProcessAsync(image, prompt);
            Console.WriteLine(result);
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