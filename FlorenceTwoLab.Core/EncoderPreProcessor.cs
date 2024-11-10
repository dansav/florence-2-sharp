using System.Diagnostics;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceTwoLab.Core;

public class EncoderPreProcessor
{
    public (DenseTensor<float> Features, DenseTensor<long> AttentionMask) Process(
        Tensor<float> visionFeatures,
        Tensor<float> textFeatures,
        IReadOnlyCollection<string> tokenized)
    {
        var projectedFeatures = ConcatenateTensors(visionFeatures, textFeatures, 1);

        var visionAttentionMask = CreateAttentionMask(Enumerable.Range(0, visionFeatures.Dimensions[1]).ToArray(), _ => 1L); 
        Debug.Assert(visionFeatures.Dimensions[1] == visionAttentionMask.Dimensions[1]);

        var textAttentionMask = CreateAttentionMask(tokenized, t => t == BartTokenizer.PadToken ? 0L : 1L);
        Debug.Assert(textFeatures.Dimensions[1] == textAttentionMask.Dimensions[1]);

        var projectedAttentionMask = ConcatenateTensors(visionAttentionMask, textAttentionMask, 1);

        return (projectedFeatures, projectedAttentionMask);
    }

    private static Tensor<TOut> CreateAttentionMask<TIn, TOut>(IReadOnlyCollection<TIn> data, Func<TIn, TOut> maskEvaluator)
    {
        var maskData = data.Select(maskEvaluator).ToArray();
        return new DenseTensor<TOut>(maskData, [1, data.Count]);
    }

    /// <summary>
    /// Concatenate two tensors along a specified axis
    /// </summary>
    /// <param name="tensor1">The first tensor to concatenate</param>
    /// <param name="tensor2">The second tensor to concatenate</param>
    /// <param name="axis">The axis along which to concatenate the tensors.</param>
    /// <typeparam name="T">The type of the tensor elements.</typeparam>
    /// <returns>
    /// The concatenated tensor.
    /// </returns>
    /// <exception cref="ArgumentException"></exception>
    private static DenseTensor<T> ConcatenateTensors<T>(Tensor<T> tensor1, Tensor<T> tensor2, int axis)
    {
        if (tensor1.Rank != tensor2.Rank)
            throw new ArgumentException("Tensors must have the same number of dimensions");

        if (axis < 0 || axis >= tensor1.Rank)
            throw new ArgumentException("Invalid axis");

        if (axis != 1)
            throw new ArgumentException("Only concatenation along axis 1 is supported");

        var newDimensions = tensor1.Dimensions.ToArray();
        newDimensions[axis] += tensor2.Dimensions[axis];

        var result = new DenseTensor<T>(newDimensions);

        // Copy data from tensor1
        for (int i = 0; i < tensor1.Length; i++)
        {
            result.SetValue(i, tensor1.GetValue(i));
        }

        // Copy data from tensor2
        var offset = (int)tensor1.Length;
        for (int i = 0; i < tensor2.Length; i++)
        {
            result.SetValue(offset + i, tensor2.GetValue(i));
        }

        return result;
    }
}