namespace FlorenceTwoLab.Core.Utils;

public class ModelHelper
{
    private readonly string _dataDir;
    private readonly HttpClient _http;

    public ModelHelper()
    {
        _dataDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache",
            "florence2lab");
        if (!Directory.Exists(_dataDir))
        {
            Directory.CreateDirectory(_dataDir);
        }

        _http = new HttpClient(new HttpClientHandler
        {
            AutomaticDecompression = System.Net.DecompressionMethods.GZip | System.Net.DecompressionMethods.Deflate
        });
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3");
    }

    public string ModelDirectory => Path.Combine(_dataDir, "models");

    public async Task EnsureModelFilesAsync()
    {
        await EnsureMetadataFilesAsync();

        var modelDir = ModelDirectory;

        string[] modelFiles =
        [
            "decoder_model.onnx",
            "embed_tokens.onnx",
            "encoder_model.onnx",
            "vision_encoder.onnx"
        ];

        foreach (var modelFile in modelFiles.Select(modelFile => Path.Combine(modelDir, modelFile)))
        {
            if (!File.Exists(modelFile))
            {
                Directory.CreateDirectory(modelDir);

                Console.WriteLine($"{Environment.NewLine}Downloading {modelFile}...");

                // TODO: allow selecting "base" or "large"
                await using var stream = await _http.GetStreamAsync(
                    $"https://huggingface.co/onnx-community/Florence-2-base/resolve/main/onnx/{Path.GetFileName(modelFile)}?download=true");
                await using var fileStream = File.Open(modelFile, FileMode.Create);
                await stream.CopyToAsync(fileStream);

                Console.WriteLine($"Download of {modelFile} completed.");
            }
        }
    }

    private async Task EnsureMetadataFilesAsync()
    {
        var metadataDir = _dataDir;

        string[] metadataFiles =
        [
            "vocab.json",
            "merges.txt"
        ];

        foreach (var metadataFile in metadataFiles)
        {
            if (!File.Exists(Path.Combine(metadataDir, metadataFile)))
            {
                Console.WriteLine($"{Environment.NewLine}Downloading metadata...");

                // https://huggingface.co/onnx-community/Florence-2-base/resolve/main/merges.txt?download=true
                await using var stream = await _http.GetStreamAsync(
                    $"https://huggingface.co/onnx-community/Florence-2-base/resolve/main/{metadataFile}?download=true");
                await using var fileStream = File.Open(Path.Combine(metadataDir, metadataFile), FileMode.Create);
                await stream.CopyToAsync(fileStream);

                Console.WriteLine("Download of metadata completed.");
            }
        }
    }
}
