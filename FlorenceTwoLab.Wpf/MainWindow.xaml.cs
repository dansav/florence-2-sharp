using System.IO;
using System.Windows;
using Florence2.Net;
using SixLabors.ImageSharp;

namespace FlorenceTwoLab;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
    }

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        var modelDir = Environment.GetEnvironmentVariable("FLORENCE2_ONNX_MODELS") ?? throw new InvalidOperationException("FLORENCE2_ONNX_MODELS environment variable is not set.");

        // var florence2 = new Florence2_Old();
        // florence2.Initialize(modelDir);

        var config = new Florence2Config
        {
            OnnxModelDirectory = modelDir,
            MetadataDirectory = System.IO.Path.Combine(modelDir, ".."),
        };
        
        var pipeline = new Florence2Pipeline(config);
        
        var testDataDir = Environment.GetEnvironmentVariable("FLORENCE2_TEST_DATA") ?? throw new InvalidOperationException("FLORENCE2_TEST_DATA environment variable is not set.");
        var image = Image.Load(Path.Combine(testDataDir, "unnamed.jpg"));

        var prompt = Florence2Tasks.CreateCaptionPrompt();

        try
        {
            var result = await pipeline.ProcessAsync(image, prompt);
            Console.WriteLine(result);
        }
        catch (Exception exception)
        {
            Console.WriteLine(exception);
            Application.Current.Shutdown();
        }
    }
}