using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FlorenceTwoLab.Core;

public class ImageProcessor
{
    private const int ImageSize = 768;
    private const float RescaleFactor = 1.0f / 255.0f;
    private static readonly float[] ImageMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] ImageStd = [0.229f, 0.224f, 0.225f];

    /// <summary>
    /// Processes the input image for model consumption
    /// </summary>
    /// <param name="image">The input image to process</param>
    /// <param name="padToSquare">
    /// Optional. Whether to pad the image to a square aspect ratio. If false, the image will be stretched. Default is true.
    /// </param>
    /// <returns>
    /// A tensor representing the processed image
    /// </returns>
    public async Task<DenseTensor<float>> ProcessImageAsync(Image image, bool padToSquare = true)
    {
        // Clone the image to avoid modifying the original
        using var processedImage = image.CloneAs<Rgb24>();

        // Resize to square (768x768)
        await ResizeImageAsync(processedImage, padToSquare);

        return await CreateNormalizedTensorAsync(processedImage);
    }

    private static async Task ResizeImageAsync(Image<Rgb24> image, bool padToSquare)
    {
        await Task.Run(() =>
        {
            image.Mutate(ctx =>
            {
                ctx.Resize(new ResizeOptions
                {
                    Size = new Size(ImageSize, ImageSize),
                    Mode = padToSquare ? ResizeMode.Pad : ResizeMode.Stretch, // Pad to maintain aspect ratio
                    PadColor = Color.Black // Use black for padding
                });
            });
        });
    }

    private static async Task<DenseTensor<float>> CreateNormalizedTensorAsync(Image<Rgb24> image)
    {
        // Create tensor with shape [count, channels, height, width]
        var tensor = new DenseTensor<float>([1, 3, ImageSize, ImageSize]);

        await Task.Run(() =>
        {
            // Process image pixels and fill tensor
            image.ProcessPixelRows(accessor =>
            {
                for (var y = 0; y < accessor.Height; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (var x = 0; x < pixelRow.Length; x++)
                    {
                        // Get RGB values
                        var pixel = pixelRow[x];

                        // Convert to float and normalize [0,1]
                        // Apply mean/std normalization
                        // Store in CHW format
                        tensor[0, 0, y, x] = (pixel.R * RescaleFactor - ImageMean[0]) / ImageStd[0]; // Red channel
                        tensor[0, 1, y, x] = (pixel.G * RescaleFactor - ImageMean[1]) / ImageStd[1]; // Green channel
                        tensor[0, 2, y, x] = (pixel.B * RescaleFactor - ImageMean[2]) / ImageStd[2]; // Blue channel
                    }
                }
            });
        });

        return tensor;
    }
}