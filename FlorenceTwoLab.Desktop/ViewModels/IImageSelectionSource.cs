using System;

using FlorenceTwoLab.Desktop.Models;

using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Desktop.ViewModels;

public interface IImageSelectionSource
{
    event Action<RegionOfInterest, Size> ImageSelectionChanged;
}
