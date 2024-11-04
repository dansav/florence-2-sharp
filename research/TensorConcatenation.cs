using System;

class TensorConcatenation
{
    static void Main()
    {
        // Example tensors with shape (2, 3, 2) and (2, 4, 2)
        int[,,] A = new int[,,]
        {
            {
                {1, 2},
                {3, 4},
                {5, 6}
            },
            {
                {7, 8},
                {9, 10},
                {11, 12}
            }
        };

        int[,,] B = new int[,,]
        {
            {
                {13, 14},
                {15, 16},
                {17, 18},
                {19, 20}
            },
            {
                {21, 22},
                {23, 24},
                {25, 26},
                {27, 28}
            }
        };

        // Concatenate along axis 1
        int[,,] result = ConcatenateAlongAxis(1, A, B);

        // Display the result
        Console.WriteLine("Result after concatenation along axis 1:");
        DisplayTensor(result);
    }

    static int[,,] ConcatenateAlongAxis(int axis, int[,,] A, int[,,] B)
    {
        // Validate tensors
        ValidateTensors(axis, A, B);

        // Get result shape
        var (dim0, dim1, dim2) = GetResultShape(axis, A, B);

        // Initialize result tensor
        int[,,] result = new int[dim0, dim1, dim2];

        // Determine if we can use block copy
        bool canUseBlockCopy = CanUseBlockCopy(axis, A);

        if (canUseBlockCopy)
        {
            // Use Buffer.BlockCopy for efficient copying
            // Calculate the number of bytes per slice to copy
            int bytesPerElement = sizeof(int);
            int elementsPerSlice = GetElementsPerSlice(axis, A);
            int bytesPerSlice = elementsPerSlice * bytesPerElement;

            // Total slices to copy
            int totalSlices = GetTotalSlices(axis, A);

            for (int slice = 0; slice < totalSlices; slice++)
            {
                // Calculate source and destination byte offsets
                int offsetA = slice * A.Length / totalSlices;
                int offsetB = slice * B.Length / totalSlices;
                int offsetResult = slice * result.Length / totalSlices;

                // Copy A's block
                Buffer.BlockCopy(A, offsetA * bytesPerElement, result, offsetResult * bytesPerElement, (A.Length / totalSlices) * bytesPerElement);

                // Copy B's block
                Buffer.BlockCopy(B, offsetB * bytesPerElement, result, (offsetResult + (A.Length / totalSlices)) * bytesPerElement, (B.Length / totalSlices) * bytesPerElement);
            }
        }
        else
        {
            // Iterate over each slice and copy elements individually
            int dim0A = A.GetLength(0);
            for (int i = 0; i < dim0A; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        if (axis == 0)
                        {
                            if (j < A.GetLength(1))
                                result[i, j, k] = A[i, j, k];
                            else
                                result[i, j, k] = B[i - A.GetLength(0), j, k];
                        }
                        else if (axis == 1)
                        {
                            if (j < A.GetLength(1))
                                result[i, j, k] = A[i, j, k];
                            else
                                result[i, j, k] = B[i, j - A.GetLength(1), k];
                        }
                        else // axis == 2
                        {
                            if (k < A.GetLength(2))
                                result[i, j, k] = A[i, j, k];
                            else
                                result[i, j, k] = B[i, j, k - A.GetLength(2)];
                        }
                    }
                }
            }
        }

        return result;
    }

    static void ValidateTensors(int axis, int[,,] A, int[,,] B)
    {
        if (axis < 0 || axis > 2)
            throw new ArgumentException("Axis must be 0, 1, or 2 for a 3D tensor.");

        for (int i = 0; i < 3; i++)
        {
            if (i != axis)
            {
                if (A.GetLength(i) != B.GetLength(i))
                    throw new ArgumentException($"Dimension {i} must match for both tensors.");
            }
        }
    }

    static (int, int, int) GetResultShape(int axis, int[,,] A, int[,,] B)
    {
        int dim0 = A.GetLength(0);
        int dim1 = A.GetLength(1);
        int dim2 = A.GetLength(2);

        switch (axis)
        {
            case 0:
                return (dim0 + B.GetLength(0), dim1, dim2);
            case 1:
                return (dim0, dim1 + B.GetLength(1), dim2);
            case 2:
                return (dim0, dim1, dim2 + B.GetLength(2));
            default:
                throw new ArgumentException("Axis must be 0, 1, or 2 for a 3D tensor.");
        }
    }

    static bool CanUseBlockCopy(int axis, int[,,] A)
    {
        // For block copy, all dimensions before the axis must be 1
        // For axis 0: no dimensions before, so always true
        // For axis 1: dim0 must be 1
        // For axis 2: dim0 and dim1 must be 1

        if (axis == 0)
            return true;
        if (axis == 1)
            return A.GetLength(0) == 1;
        if (axis == 2)
            return A.GetLength(0) == 1 && A.GetLength(1) == 1;
        return false;
    }

    static int GetElementsPerSlice(int axis, int[,,] A)
    {
        // Number of elements to copy in one block per slice
        switch (axis)
        {
            case 0:
                // Entire A or B along axis 0
                return A.GetLength(1) * A.GetLength(2);
            case 1:
                // Along axis 1
                return A.GetLength(2);
            case 2:
                // Along axis 2
                return 1;
            default:
                throw new ArgumentException("Axis must be 0, 1, or 2 for a 3D tensor.");
        }
    }

    static int GetTotalSlices(int axis, int[,,] A)
    {
        // Number of slices to copy
        switch (axis)
        {
            case 0:
                return 1;
            case 1:
                return A.GetLength(0);
            case 2:
                return A.GetLength(0) * A.GetLength(1);
            default:
                throw new ArgumentException("Axis must be 0, 1, or 2 for a 3D tensor.");
        }
    }

    static void DisplayTensor(int[,,] tensor)
    {
        for (int i = 0; i < tensor.GetLength(0); i++)
        {
            Console.WriteLine($"Slice {i}:");
            for (int j = 0; j < tensor.GetLength(1); j++)
            {
                Console.Write("[");
                for (int k = 0; k < tensor.GetLength(2); k++)
                {
                    Console.Write(tensor[i, j, k] + (k < tensor.GetLength(2) - 1 ? ", " : ""));
                }
                Console.WriteLine("]");
            }
            Console.WriteLine();
        }
    }
}
