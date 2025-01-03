using SixLabors.ImageSharp;

namespace FlorenceTwoLab.Core;

/// <summary>
/// Handles converting between image coordinates and Florence-2 location tokens
/// </summary>
public static class Florence2LocationTokens
{
    // Florence uses coordinates from 0-999 for location tokens
    private const int TokenCoordinateRange = 1000;

    /// <summary>
    /// Creates a normalized location token string from a region and image size
    /// </summary>
    public static string CreateNormalizedRegionString(this RectangleF region)
    {
        var topleft = new Point(
            (int) (region.Left * TokenCoordinateRange),
            (int) (region.Top * TokenCoordinateRange));
        var bottomRight = new Point(
            (int) (region.Right * TokenCoordinateRange),
            (int) (region.Bottom * TokenCoordinateRange));
        
        return CoordinatesToTokens(topleft, bottomRight);
    }
    
    
    /// <summary>
    /// Converts pixel coordinates to location token strings
    /// </summary>
    public static string CoordinatesToTokens(Point topLeft, Point bottomRight)
    {
        // Florence expects coordinates in format: <loc_x1><loc_y1><loc_x2><loc_y2>
        return $"<loc_{NormalizeCoordinate(topLeft.X)}>" +
               $"<loc_{NormalizeCoordinate(topLeft.Y)}>" +
               $"<loc_{NormalizeCoordinate(bottomRight.X)}>" +
               $"<loc_{NormalizeCoordinate(bottomRight.Y)}>";
    }

    /// <summary>
    /// Converts bounding box coordinates to location token strings
    /// </summary>
    public static string CoordinatesToTokens(Rectangle boundingBox)
    {
        return CoordinatesToTokens(
            new Point(boundingBox.Left, boundingBox.Top),
            new Point(boundingBox.Right, boundingBox.Bottom));
    }

    /// <summary>
    /// Converts location tokens back to pixel coordinates
    /// </summary>
    public static Rectangle TokensToCoordinates(string locationTokens, Size imageSize)
    {
        var coordinates = ParseLocationTokens(locationTokens);
        if (coordinates.Count != 4)
            throw new ArgumentException("Location tokens must contain exactly 4 coordinates", nameof(locationTokens));

        return new Rectangle(
            DenormalizeCoordinate(coordinates[0], imageSize.Width),
            DenormalizeCoordinate(coordinates[1], imageSize.Height),
            DenormalizeCoordinate(coordinates[2], imageSize.Width) - DenormalizeCoordinate(coordinates[0], imageSize.Width),
            DenormalizeCoordinate(coordinates[3], imageSize.Height) - DenormalizeCoordinate(coordinates[1], imageSize.Height)
        );
    }

    /// <summary>
    /// Overload for single coordinate normalization when dimension is unknown
    /// </summary>
    private static int NormalizeCoordinate(int coordinate)
    {
        return Math.Clamp(coordinate, 0, TokenCoordinateRange - 1);
    }

    /// <summary>
    /// Converts a normalized coordinate back to pixel space
    /// </summary>
    public static int DenormalizeCoordinate(int normalizedCoordinate, int imageDimension)
    {
        return (normalizedCoordinate * imageDimension) / TokenCoordinateRange;
    }

    /// <summary>
    /// Extracts coordinate values from location tokens
    /// </summary>
    private static List<int> ParseLocationTokens(string tokens)
    {
        var coordinates = new List<int>();
        var matches = System.Text.RegularExpressions.Regex.Matches(tokens, @"<loc_(\d+)>");
        
        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            if (int.TryParse(match.Groups[1].Value, out int coordinate))
            {
                coordinates.Add(coordinate);
            }
        }

        return coordinates;
    }
}
