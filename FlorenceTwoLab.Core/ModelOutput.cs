using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceTwoLab.Core;

public class ModelOutput : IDisposable
{
    private readonly IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _outputs;
    
    public ModelOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        _outputs = outputs;
    }

    public Tensor<float> GetLogits()
    {
        return _outputs.First(o => o.Name == "logits").AsTensor<float>();
    }

    public IReadOnlyList<Tensor<float>> GetPresent()
    {
        var presentTensors = new List<Tensor<float>>();
        
        foreach (var output in _outputs)
        {
            if (output.Name.StartsWith("present."))
            {
                presentTensors.Add(output.AsTensor<float>());
            }
        }

        return presentTensors;
    }

    public void Dispose()
    {
        _outputs?.Dispose();
    }
}