using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Florence2.Net;

public class Florence2ImageProcessor
{
    private readonly Florence2Config _config;

    public Florence2ImageProcessor(Florence2Config config)
    {
        _config = config;
    }

    /// <summary>
    /// Processes an image according to Florence-2 requirements
    /// </summary>
    public async Task<DenseTensor<float>> ProcessImageAsync(Image image)
    {
        // Clone the image to avoid modifying the original
        using var processedImage = image.CloneAs<Rgb24>();
        
        // Florence-2 expects:
        // 1. Resize to square (768x768)
        // 2. Convert to float [0,1]
        // 3. Normalize using mean/std
        // 4. Convert to NCHW format (batch, channels, height, width)
        
        await ResizeImageAsync(processedImage);
        return await CreateNormalizedTensorAsync(processedImage);
    }

    /// <summary>
    /// Resizes image to model dimensions while preserving aspect ratio
    /// </summary>
    private async Task ResizeImageAsync(Image<Rgb24> image)
    {
        await Task.Run(() =>
        {
            image.Mutate(ctx =>
            {
                ctx.Resize(new ResizeOptions
                {
                    Size = new Size(_config.ImageSize, _config.ImageSize),
                    Mode = ResizeMode.Pad, // Pad to maintain aspect ratio
                    PadColor = Color.Black // Use black for padding
                });
            });
        });
    }

    /// <summary>
    /// Creates a normalized tensor in NCHW format from the processed image
    /// </summary>
    private async Task<DenseTensor<float>> CreateNormalizedTensorAsync(Image<Rgb24> image)
    {
        // Create tensor with shape [1, 3, height, width]
        var tensor = new DenseTensor<float>(new[] { 1, 3, _config.ImageSize, _config.ImageSize });
        
        await Task.Run(() =>
        {
            // Process image pixels and fill tensor
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelRow = accessor.GetRowSpan(y);
                    
                    for (int x = 0; x < pixelRow.Length; x++)
                    {
                        // Get RGB values
                        var pixel = pixelRow[x];
                        
                        // Convert to float and normalize [0,1]
                        float r = pixel.R * _config.RescaleFactor;
                        float g = pixel.G * _config.RescaleFactor;
                        float b = pixel.B * _config.RescaleFactor;
                        
                        // Apply mean/std normalization
                        r = (r - _config.ImageMean[0]) / _config.ImageStd[0];
                        g = (g - _config.ImageMean[1]) / _config.ImageStd[1];
                        b = (b - _config.ImageMean[2]) / _config.ImageStd[2];
                        
                        // Store in CHW format
                        tensor[0, 0, y, x] = r; // Red channel
                        tensor[0, 1, y, x] = g; // Green channel
                        tensor[0, 2, y, x] = b; // Blue channel
                    }
                }
            });
        });
        
        return tensor;
    }

    /// <summary>
    /// Processes multiple images in a batch
    /// </summary>
    public async Task<DenseTensor<float>> ProcessImagesAsync(IReadOnlyList<Image> images)
    {
        if (images.Count == 0)
            throw new ArgumentException("At least one image is required", nameof(images));

        // Create tensor for batch [batch_size, 3, height, width]
        var tensor = new DenseTensor<float>(new[] { images.Count, 3, _config.ImageSize, _config.ImageSize });
        
        // Process each image
        for (int i = 0; i < images.Count; i++)
        {
            using var processedImage = images[i].CloneAs<Rgb24>();
            await ResizeImageAsync(processedImage);
            
            // Fill tensor for this batch item
            await Task.Run(() =>
            {
                processedImage.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < accessor.Height; y++)
                    {
                        Span<Rgb24> pixelRow = accessor.GetRowSpan(y);
                        
                        for (int x = 0; x < pixelRow.Length; x++)
                        {
                            var pixel = pixelRow[x];
                            
                            // Normalize and store in batch tensor
                            float r = ((pixel.R * _config.RescaleFactor) - _config.ImageMean[0]) / _config.ImageStd[0];
                            float g = ((pixel.G * _config.RescaleFactor) - _config.ImageMean[1]) / _config.ImageStd[1];
                            float b = ((pixel.B * _config.RescaleFactor) - _config.ImageMean[2]) / _config.ImageStd[2];
                            
                            tensor[i, 0, y, x] = r;
                            tensor[i, 1, y, x] = g;
                            tensor[i, 2, y, x] = b;
                        }
                    }
                });
            });
        }
        
        return tensor;
    }
}

public static class Florence2ImageProcessorExtensions
{
    /// <summary>
    /// Converts a byte array to an Image that can be processed
    /// </summary>
    public static async Task<Image> LoadImageFromBytesAsync(this Florence2ImageProcessor _, byte[] imageData)
    {
        using var stream = new MemoryStream(imageData);
        return await Image.LoadAsync(stream);
    }
    
    /// <summary>
    /// Loads an image from a file path
    /// </summary>
    public static async Task<Image> LoadImageFromFileAsync(this Florence2ImageProcessor _, string filePath)
    {
        return await Image.LoadAsync(filePath);
    }
}