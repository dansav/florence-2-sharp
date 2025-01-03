using System.Collections.ObjectModel;

using Avalonia;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

using FlorenceTwoLab.Desktop.Models;

namespace FlorenceTwoLab.Desktop.ViewModels;

public partial class ImageRegionSelectorViewModel : ObservableObject
{
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsDrawing))]
    [NotifyPropertyChangedFor(nameof(DrawingLeft))]
    [NotifyPropertyChangedFor(nameof(DrawingTop))]
    [NotifyPropertyChangedFor(nameof(DrawingWidth))]
    [NotifyPropertyChangedFor(nameof(DrawingHeight))]
    private RegionOfInterest? _currentRegion;
    
    public bool IsDrawing => CurrentRegion is not null;
    
    public double DrawingLeft => CurrentRegion?.Left ?? 0;
    
    public double DrawingTop => CurrentRegion?.Top ?? 0;
    
    public double DrawingWidth => CurrentRegion?.Width ?? 0;
    
    public double DrawingHeight => CurrentRegion?.Height ?? 0;
    
    public ObservableCollection<RegionOfInterest> Regions { get; } = new();
    
    public bool HasRegions => Regions.Count > 0;
    
    public void ClearRegions()
    {
        Regions.Clear();
        OnPropertyChanged(nameof(HasRegions));
    }

    [RelayCommand]
    private void StartDrawing(Point point)
    {
        Regions.Clear();

        var region = new RegionOfInterest();
        region.Start(point);
        
        CurrentRegion = region;
    }
    
    [RelayCommand]
    private void UpdateDrawing(Point point)
    {
        if (CurrentRegion is null) return;
        
        CurrentRegion.Update(point);
        OnPropertyChanged(nameof(IsDrawing));
        OnPropertyChanged(nameof(DrawingLeft));
        OnPropertyChanged(nameof(DrawingTop));
        OnPropertyChanged(nameof(DrawingWidth));
        OnPropertyChanged(nameof(DrawingHeight));
    }
    
    [RelayCommand]
    private void FinishDrawing(Rect? imageBounds)
    {
        if (CurrentRegion is null) return;
        
        CurrentRegion.End(imageBounds);
        if (CurrentRegion.Width > 0 && CurrentRegion.Height > 0)
        {
            Regions.Add(CurrentRegion);
            OnPropertyChanged(nameof(HasRegions));
        }
        CurrentRegion = null;
    }
}
