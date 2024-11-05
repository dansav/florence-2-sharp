using Microsoft.ML.OnnxRuntime;

namespace FlorenceTwoLab.Core;

public class Florence2Debug
{
    public void DocumentInputsAndOutputs(string? modelDir)
    {
        if (string.IsNullOrEmpty(modelDir))
        {
            throw new ArgumentNullException(nameof(modelDir));
        }

        string[] models = [
            "decoder_model.onnx",
            "embed_tokens.onnx",
            "encoder_model.onnx",
            "vision_encoder.onnx"
        ];
        
        foreach (var model in models)
        {
            var modelPath = Path.Combine(modelDir, model);
            Console.WriteLine($"\r\n### {model}:");
            
            var session = new InferenceSession(modelPath);
            
            Console.WriteLine("_Inputs:_\r\n```yaml");
            foreach (var (name, value) in session.InputMetadata)
            {
                Console.WriteLine($"- name: {name}");
                Console.WriteLine($"  value: {value.ElementDataType}[{string.Join(", ", value.Dimensions.Zip(value.SymbolicDimensions, (d, s) => string.IsNullOrEmpty(s) ? d.ToString() : s))}]");
            }
            
            Console.WriteLine("```\r\n\r\n_Outputs:_\r\n```yaml");
            foreach (var (name, value) in session.OutputMetadata)
            {
                Console.WriteLine($"- name: {name}");
                Console.WriteLine($"  value: {value.ElementDataType}[{string.Join(", ", value.Dimensions.Zip(value.SymbolicDimensions, (d, s) => string.IsNullOrEmpty(s) ? d.ToString() : s))}]");
            }
            Console.WriteLine("```");
        }
    }
}