namespace FlorenceTwoLab.Core.Utils;

public class DataHelper
{
    private readonly string _dataDir;
    private readonly HttpClient _http;

    public DataHelper()
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

    public string TestDataDirectory => Path.Combine(_dataDir, "test_data");

    public async Task EnsureTestDataFilesAsync()
    {
        var url =
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true";

        var testDataDir = TestDataDirectory;

        if (!File.Exists(Path.Combine(testDataDir, "car.jpg")))
        {
            Directory.CreateDirectory(testDataDir);

            Console.WriteLine($"{Environment.NewLine}Downloading test data...");

            await using var stream = await _http.GetStreamAsync(url);
            await using var fileStream = File.Open(Path.Combine(testDataDir, "car.jpg"), FileMode.Create);
            await stream.CopyToAsync(fileStream);

            Console.WriteLine("Download of test data completed.");
        }
    }
}