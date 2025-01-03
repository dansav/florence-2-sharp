using System;

using Avalonia;

using SixLabors.ImageSharp;

using Point = Avalonia.Point;
using Size = Avalonia.Size;

namespace FlorenceTwoLab.Desktop.Models;

public class RegionOfInterest
{
    private Point _startPoint;

    private Point _endPoint;
    
    public double Width => Math.Abs(_endPoint.X - _startPoint.X);

    public double Height => Math.Abs(_endPoint.Y - _startPoint.Y);

    public double Left => Math.Min(_startPoint.X, _endPoint.X);
    
    public double Top => Math.Min(_startPoint.Y, _endPoint.Y);
    
    public RectangleF RelativeBounds { get; private set; }

    public void Start(Point point)
    {
        _startPoint = point;
        _endPoint = point;
    }

    public void Update(Point point)
    {
        _endPoint = point;
    }
    
    public void End(Rect? imageBounds)
    {
        if (imageBounds is null) return;

        // Ensure the region is within the image bounds
        var left = Math.Max(Left, imageBounds.Value.Left);
        var top = Math.Max(Top, imageBounds.Value.Top);
        var right = Math.Min(Left + Width, imageBounds.Value.Right);
        var bottom = Math.Min(Top + Height, imageBounds.Value.Bottom);

        // Update the region with the new bounds
        _startPoint = new Point(left, top);
        _endPoint = new Point(right, bottom);
        
        // Calculate the relative bounds used in non-UI code
        RelativeBounds = CalculateRelativeBounds(imageBounds.Value.Position, imageBounds.Value.Size);
    }
    
    private RectangleF CalculateRelativeBounds(Point position, Size imageSize)
    {
        var left = (Left - position.X) / imageSize.Width;
        var top = (Top - position.Y) / imageSize.Height;
        var width = Width / imageSize.Width;
        var height = Height / imageSize.Height;
        
        return new RectangleF((float)left, (float)top, (float)width, (float)height);
    }
}
