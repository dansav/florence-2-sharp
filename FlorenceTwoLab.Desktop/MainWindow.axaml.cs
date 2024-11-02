using System;
using System.IO;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Interactivity;
using Florence2.Net;
using FlorenceTwoLab.Core.Utils;

namespace FlorenceTwoLab.Desktop;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        Console.SetOut(new ConsoleHelper(Console.Out, ScreenConsole));

        Loaded += MainWindow_Loaded;
    }

    private async void MainWindow_Loaded(object? sender, RoutedEventArgs e)
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
            OnnxModelDirectory = modelDir,
            MetadataDirectory = System.IO.Path.Combine(modelDir, ".."),
        };
        
        var pipeline = new Florence2Pipeline(config);

        string? testDataDir = Environment.GetEnvironmentVariable("FLORENCE2_TEST_DATA");
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