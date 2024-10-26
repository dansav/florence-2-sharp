namespace Florence2.Net;

public sealed class Florence2Config
{
    public string OnnxModelDirectory { get; set; } = "Models";
    public string MetadataDirectory { get; set; } = "Utils";
    
    public int ImageSize { get; set; } = 768;
    public float[] ImageMean { get; set; } = new[] { 0.485f, 0.456f, 0.406f };
    public float[] ImageStd { get; set; } = new[] { 0.229f, 0.224f, 0.225f };
    public float RescaleFactor { get; set; } = 1.0f / 255.0f;
    public int VocabSize { get; set; } = 51289;
    public int ProjectionDim { get; set; } = 768;
}