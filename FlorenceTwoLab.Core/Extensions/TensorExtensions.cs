using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceTwoLab.Core.Extensions;

public static class TensorExtensions
{
    /// <summary>
    /// Concatenate two tensors along a specified axis
    /// </summary>
    /// <param name="first"> 
    /// The first tensor to concatenate 
    /// </param>
    /// <param name="second">
    /// The second tensor to concatenate
    /// </param>
    /// <param name="axis">
    /// The axis along which to concatenate the tensors. Default is 0.
    /// </param>
    /// <typeparam name="T">
    /// The type of the tensor elements.
    /// </typeparam>
    /// <returns>
    /// The concatenated tensor.
    /// </returns>
    public static Tensor<T> Concatenate<T>(this Tensor<T> first, Tensor<T> second, int axis = 0)
    {
        if (first.Rank != second.Rank)
        {
            throw new ArgumentException("Tensors must have the same rank (number of dimensions).");
        }

        if (axis < 0 || axis >= first.Dimensions.Length)
        {
            throw new ArgumentException("Invalid axis.");
        }
        
        for (var i = 0; i < first.Dimensions.Length; i++)
        {
            if (i != axis && first.Dimensions[i] != second.Dimensions[i])
            {
                throw new ArgumentException("Tensors must have the same dimensions except for the concatenation axis.");
            }
        }

        var newDimensions = new int[first.Dimensions.Length];
        newDimensions[axis] += second.Dimensions[axis];
        
        var result = new DenseTensor<T>(newDimensions);
        
        // Can we use flat copy?
        if (axis == 0 || newDimensions.Take(axis).All(d => d == 1))
        {
            var j = 0;
            // Copy data from tensor1
            for (int i = 0; i < first.Length; i++)
            {
                result[j++] = first[i];
            }

            // Copy data from tensor2
            for (int i = 0; i < second.Length; i++)
            {
                result[j++] = second[i];
            }
        }
        else
        {
            throw new NotImplementedException("All dimensions before the concatenation axis must be 1.");
        }

        return result;
    }
}