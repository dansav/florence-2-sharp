using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;

using FlorenceTwoLab.Desktop.ViewModels;

namespace FlorenceTwoLab.Desktop.Views;

public partial class ImageRegionSelectorView : UserControl
{
    private Image? _image;
    public static readonly DirectProperty<ImageRegionSelectorView, Image?> ImageProperty = AvaloniaProperty.RegisterDirect<ImageRegionSelectorView, Image?>("Image", o => o.Image, (o, v) => o.Image = v);

    public ImageRegionSelectorView()    
    {
        InitializeComponent();
    }

    public Image? Image
    {
        get { return _image; }
        set { SetAndRaise(ImageProperty, ref _image, value); }
    }

    private void OnPointerPressed(object sender, PointerPressedEventArgs e)
    {
        if (DataContext is ImageRegionSelectorViewModel vm)
        {
            var point = e.GetPosition(DrawingCanvas);
            vm.StartDrawingCommand.Execute(point);
        }
    }

    private void OnPointerMoved(object sender, PointerEventArgs e)
    {
        if (DataContext is ImageRegionSelectorViewModel vm)
        {
            var point = e.GetPosition(DrawingCanvas);
            vm.UpdateDrawingCommand.Execute(point);
        }
    }

    private void OnPointerReleased(object sender, PointerReleasedEventArgs e)
    {
        if (DataContext is ImageRegionSelectorViewModel vm)
        {
            var imageBounds = GetImageBoundsRelativeToCanvas();
            vm.FinishDrawingCommand.Execute(imageBounds);
        }
    }
    
    public Rect? GetImageBoundsRelativeToCanvas()
    {
        if (Image == null || DrawingCanvas == null)
        {
            return null;
        }
            
        var imageBounds = TransformBoundsTo(Image, DrawingCanvas);
        return imageBounds;
    }
    
    private static Rect? TransformBoundsTo( Visual visual, Visual relativeTo)
    {
        if (visual.TransformToVisual(relativeTo) is { } transformation)
        {
            var p0 = transformation.Transform(default);
            var p1 = transformation.Transform(new(visual.Bounds.Width, visual.Bounds.Height));
            return new(p0, p1);
        }
        return default;
    }
}


