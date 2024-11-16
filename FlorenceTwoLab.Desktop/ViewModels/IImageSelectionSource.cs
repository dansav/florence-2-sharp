using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Desktop.ViewModels;

public interface IImageSelectionSource
{
    Rectangle Selection { get; }
    Size ImageSize { get; }
}