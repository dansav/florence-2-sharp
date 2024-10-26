using System.Text.Json;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FlorenceTwoLab.Model;

public class AutoProcessor
{
    private const string preprocessor_config_json = """
                                                              {
                                                                "auto_map": {
                                                                  "AutoProcessor": "processing_florence2.Florence2Processor"
                                                                 },
                                                                "_valid_processor_keys": [
                                                                  "images",
                                                                  "do_resize",
                                                                  "size",
                                                                  "resample",
                                                                  "do_rescale",
                                                                  "rescale_factor",
                                                                  "do_normalize",
                                                                  "image_mean",
                                                                  "image_std",
                                                                  "return_tensors",
                                                                  "data_format",
                                                                  "input_data_format",
                                                                  "do_convert_rgb"
                                                                ],
                                                                "do_convert_rgb": null,
                                                                "do_normalize": true,
                                                                "do_rescale": true,
                                                                "do_resize": true,
                                                                "do_center_crop": false,
                                                                "image_processor_type": "CLIPImageProcessor",
                                                                "image_seq_length": 577,
                                                                "image_mean": [0.485, 0.456, 0.406],
                                                                "image_std":  [0.229, 0.224, 0.225],
                                                                "processor_class": "Florence2Processor",
                                                                "resample": 3,
                                                                "size": {
                                                                  "height": 768,
                                                                  "width":768 
                                                                },
                                                                "crop_size": {
                                                                  "height": 768,
                                                                  "width": 768
                                                                }
                                                              }                                                          
                                                              """;

    private JsonElement _config;

    public Task InitializeAsync()
    {
        // Initialize image processing parameters
        _config = JsonSerializer.Deserialize<JsonElement>(preprocessor_config_json);
        return Task.CompletedTask;
    }

    public Tensor<float> PreprocessImage(Image inputImage)
    {
        var image = inputImage.CloneAs<Rgba32>();
        
        // Resize
        if (_config.GetProperty("do_resize").GetBoolean())
        {
            var size = _config.GetProperty("size");
            int width = size.GetProperty("width").GetInt32();
            int height = size.GetProperty("height").GetInt32();
            image.Mutate(x => x.Resize(width, height));
        }


        // Convert to float array and apply preprocessing
        var pixels = new float[3 * image.Width * image.Height];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < pixelRow.Length; x++)
                {
                    int idx = (y * accessor.Width + x) * 3;
                    pixels[idx] = pixelRow[x].R / 255f;
                    pixels[idx + 1] = pixelRow[x].G / 255f;
                    pixels[idx + 2] = pixelRow[x].B / 255f;
                }
            }
        });

        // Rescale
        if (_config.GetProperty("do_rescale").GetBoolean())
        {
            float rescaleFactor = 1f / 255f; // Assuming default rescale factor
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] *= rescaleFactor;
            }
        }

        // Normalize
        if (_config.GetProperty("do_normalize").GetBoolean())
        {
            var mean = _config.GetProperty("image_mean").EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
            var std = _config.GetProperty("image_std").EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
            for (int i = 0; i < pixels.Length; i++)
            {
                int channel = i % 3;
                pixels[i] = (pixels[i] - mean[channel]) / std[channel];
            }
        }

        // Create and return tensor
        return new DenseTensor<float>(pixels, new[]
        {
            1,
            3,
            image.Height,
            image.Width
        });
    }
}