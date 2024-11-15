# What is a QuadBox?

A QuadBox (quadrilateral box) and a rectangle are different geometric shapes:

Rectangle:

- Has 4 corners/points, all at 90-degree angles
- Can be defined by just 2 points (top-left and bottom-right)
- Is always axis-aligned (horizontal and vertical lines)
- Defined with x, y, width, height properties

QuadBox (Quadrilateral):

- Has 4 corners/points that can be at any angle
- Needs all 4 points to be defined (8 coordinates: x1,y1,x2,y2,x3,y3,x4,y4)
- Can represent rotated, skewed, or perspective-distorted text
- Points typically go clockwise starting from top-left

Here's a visual representation:

```text
Rectangle:                     QuadBox/Quadrilateral:
   A--------B                    A-------B
   |        |                   /       /
   |        |                  /       /
   |        |                 /       /
   D--------C                D-------C

Rectangle properties:          QuadBox coordinates:
x = A.x                       Points[0,1] = A.x,A.y  
y = A.y                       Points[2,3] = B.x,B.y
width = C.x - A.x             Points[4,5] = C.x,C.y
height = C.y - A.y            Points[6,7] = D.x,D.y
```

The key difference is that QuadBoxes are used for OCR because text in images can be:

- Rotated (like on a sign photographed at an angle)
- Skewed (perspective distortion from camera angle)
- At arbitrary angles

A Rectangle can't represent these cases accurately since it must remain axis-aligned. That's why the OCR output provides 8 coordinates - it needs to specify the exact position of each corner of the text region.

When we have to convert a QuadBox to a Rectangle (like in our code), we essentially create a bounding box that encompasses the entire quadrilateral by finding the minimum and maximum x,y coordinates. This loses some precision but is sometimes necessary for compatibility with APIs that only work with rectangles.

Here's an example of why this matters for OCR:

```text
Rectangle (less accurate):          QuadBox (more accurate):
   +------------+                     A-------B
   |  Angled    |                   /  Angled /
   |    Text    |                  /   Text  /
   |            |                 /         /
   +------------+                D--------C
```

The QuadBox precisely outlines the text region, while the Rectangle has to be larger to contain the entire text, including empty space.

This is particularly important in scenarios like:

- Reading text from signs photographed at an angle
- Processing documents that aren't perfectly aligned
- Reading text that follows curves or non-rectangular paths
- Handling perspective distortion in images

That's why the OCR output from Florence-2 provides 8 coordinates per text region instead of just 4 - it preserves the exact shape and orientation of the detected text regions.
