using System;

using Avalonia;

namespace FlorenceTwoLab.Desktop.Models;

public class RegionOfInterest
{
    public Point StartPoint { get; set; }
    
    public Point EndPoint { get; set; }
    
    public bool IsComplete { get; set; }

    public double Width => Math.Abs(EndPoint.X - StartPoint.X);

    public double Height => Math.Abs(EndPoint.Y - StartPoint.Y);

    public double Left => Math.Min(StartPoint.X, EndPoint.X);
    
    public double Top => Math.Min(StartPoint.Y, EndPoint.Y);

    public void Crop(Rect? imageBounds)
    {
        if (imageBounds is null) return;
        
        var left = Math.Max(Left, imageBounds.Value.Left);
        var top = Math.Max(Top, imageBounds.Value.Top);
        var right = Math.Min(Left + Width, imageBounds.Value.Right);
        var bottom = Math.Min(Top + Height, imageBounds.Value.Bottom);
        
        StartPoint = new Point(left, top);
        EndPoint = new Point(right, bottom);
    }
}
