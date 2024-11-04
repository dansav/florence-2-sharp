namespace Florence2.Net;

public sealed class Florence2Config
{
    public string OnnxModelDirectory { get; } = "Models";
    public string MetadataDirectory { get; } = "Utils";

    public int ImageSize { get; } = 768;
    public float[] ImageMean { get; } = [0.485f, 0.456f, 0.406f];
    public float[] ImageStd { get; } = [0.229f, 0.224f, 0.225f];
    public float RescaleFactor { get; } = 1.0f / 255.0f;
    public int VocabSize { get; } = 51289;
    public int ProjectionDim { get; } = 768;
}