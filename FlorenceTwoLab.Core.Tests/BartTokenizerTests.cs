using FlorenceTwoLab.Core.Tests.Resources;

using FluentAssertions;

namespace FlorenceTwoLab.Core.Tests;

public class BartTokenizerTests
{
    private static Task<BartTokenizer> CreateTokenizerAsync()
    {
        var merges = Data.Merges;
        var vocabJson = Data.VocabJson;
        var addedTokensJson = Data.AddedTokensJson;
        return BartTokenizer.FromPretrainedAsync(merges, vocabJson, addedTokensJson);
    }

    [Fact]
    public async Task Tokenize_PlainText_ValidTokens()
    {
        // Arrange
        var tokenizer = await CreateTokenizerAsync();
        var text = "Hello, world!";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        tokens.Should().BeEquivalentTo("Hello", ",", "Ġworld", "!");
    }
    
    [Fact]
    public async Task Tokenize_WithSpecialTokens_CorrectTokens()
    {
        // Arrange
        var tokenizer = await CreateTokenizerAsync();
        var text = "What is the region <loc_123><loc_234><loc_345><loc_456>?";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        tokens.Should().BeEquivalentTo("What", "Ġis", "Ġthe", "Ġregion", "Ġ", "<loc_123>", "<loc_234>", "<loc_345>", "<loc_456>", "?");
    }
}
