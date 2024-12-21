namespace FlorenceTwoLab.Core.Tests.Resources;

public class Data
{
    public static Stream Merges => GetEmbeddedFileStream("merges.txt");
    
    public static Stream VocabJson => GetEmbeddedFileStream("vocab.json");
    
    public static Stream AddedTokensJson => GetEmbeddedFileStream("added_tokens.json");
    
    private static Stream GetEmbeddedFileStream(string name, string prefix = "FlorenceTwoLab.Core.Tests.Resources.")
    {
        var assembly = typeof(Data).Assembly;
        var stream = assembly.GetManifestResourceStream(prefix + name);
        if (stream == null) throw new FileNotFoundException("Embedded resource not found: " + name);
        return stream;
    }
}
