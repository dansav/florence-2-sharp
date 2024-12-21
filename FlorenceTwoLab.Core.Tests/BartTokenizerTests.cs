using FlorenceTwoLab.Core.Tests.Resources;

namespace FlorenceTwoLab.Core.Tests;

public class BartTokenizerTests
{

    public BartTokenizerTests()
    {
    var  merges = Data.Merges;
    var _vocabJ§son = Data.VocabJson;
    var _addedTokensJson = Data.AddedTokensJson;
        _sut = new BartTokenizer(merges, _vocabJson, _addedTokensJson);
    }
    
    [Fact]
    public void Tokenize_PlainText_ValidTokens()
    {
        // Arrange
        var tokenizer = new BartTokenizer();
        var text = "Hello, world!";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.Equal(new[] { "Hello", ",", "world", "!" }, tokens);
    }
}
