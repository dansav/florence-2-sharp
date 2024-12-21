namespace FlorenceTwoLab.Core.Tests.Resources;

public class Data
{
    public static string Merges => GetEmbeddedFileAsString("merges.txt");
    
    public static string VocabJson => GetEmbeddedFileAsString("vocab.json");
    
    public static string AddedTokensJson => GetEmbeddedFileAsString("added_tokens.json");
    
    private static string GetEmbeddedFileAsString(string name, string prefix = "FlorenceTwoLab.Core.Tests.Resources.")
    {
        var assembly = typeof(Data).Assembly;
        using var stream = assembly.GetManifestResourceStream(prefix + name);
        if (stream == null) throw new FileNotFoundException("Embedded resource not found: " + name);

        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }
}
