using System;
using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media.Imaging;
using SixLabors.ImageSharp;
using System.IO;

namespace FlorenceTwoLab.Desktop;

public class ImageConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is Image imageSharpImage)
        {
            using (var memoryStream = new MemoryStream())
            {
                imageSharpImage.SaveAsBmp(memoryStream);
                memoryStream.Seek(0, SeekOrigin.Begin);
                return new Bitmap(memoryStream);
            }
        }
        return null;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}