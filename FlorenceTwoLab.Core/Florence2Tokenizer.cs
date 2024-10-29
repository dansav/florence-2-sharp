﻿using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Florence2.Net;

public class Florence2Tokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _reverseVocab;
    private readonly List<(string First, string Second)> _bpeMerges;
    private readonly Dictionary<string, int> _specialTokens;
    
    private const int BosTokenId = 0;  // <s>
    private const int EosTokenId = 2;  // </s>
    private const int PadTokenId = 1;  // <pad>
    private const int UnkTokenId = 3;  // <unk>

    public Florence2Tokenizer(Florence2Config config)
    {
        var vocabPath = Path.Combine(config.MetadataDirectory, "vocab.json");
        var mergesPath = Path.Combine(config.MetadataDirectory, "merges.txt");
        
        _vocab = LoadVocabulary(vocabPath);
        _reverseVocab = CreateReverseVocabulary(_vocab);
        _bpeMerges = LoadMerges(mergesPath);
        _specialTokens = InitializeSpecialTokens();
    }
    
    /// <summary>
    /// Encodes text into token IDs
    /// </summary>
    public async Task<Tensor<long>> EncodeAsync(string text)
    {
        // Pre-process text and split into words
        var words = await PreProcessTextAsync(text);
        
        // Tokenize each word using BPE
        var tokens = new List<int>();
        tokens.Add(BosTokenId); // Add beginning of sequence token
        
        foreach (var word in words)
        {
            tokens.AddRange(EncodeWord(word));
        }
        
        tokens.Add(EosTokenId); // Add end of sequence token
        
        // Convert to tensor
        return CreateTensor(tokens);
    }
    
    /// <summary>
    /// Decode a sequence of token IDs back to text
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var result = new StringBuilder();
        var needsPrefix = true; // Track if we need a space prefix

        foreach (var tokenId in tokenIds)
        {
            // Skip special tokens in final output
            if (tokenId == BosTokenId || tokenId == EosTokenId || tokenId == PadTokenId)
                continue;

            var token = DecodeSingleToken(tokenId);

            // Handle special tokens
            if (_specialTokens.ContainsValue(tokenId))
            {
                result.Append(token);
                needsPrefix = true;
                continue;
            }

            // Check if token starts with Ġ (space indicator in byte-level BPE)
            if (token.StartsWith("Ġ"))
            {
                result.Append(' ').Append(token[1..]);
                needsPrefix = false;
            }
            else
            {
                if (needsPrefix && result.Length > 0)
                    result.Append(' ');
                result.Append(token);
                needsPrefix = false;
            }
        }

        return result.ToString().Trim();
    }
    
    /// <summary>
    /// Decode logits tensor to text by taking the highest probability token at each position
    /// </summary>
    public string DecodeLogits(Tensor<float> logits)
    {
        var tokens = new List<int>();

        // Process each position in the sequence
        for (int pos = 0; pos < logits.Dimensions[1]; pos++)
        {
            float maxProb = float.MinValue;
            int maxToken = 0;

            // Find highest probability token
            for (int token = 0; token < logits.Dimensions[2]; token++)
            {
                var prob = logits[0, pos, token];
                if (prob > maxProb)
                {
                    maxProb = prob;
                    maxToken = token;
                }
            }

            // Stop at end of sequence token
            if (maxToken == EosTokenId)
                break;

            tokens.Add(maxToken);
        }

        return Decode(tokens.ToArray());
    }
    
    /// <summary>
    /// Decode a tensor of token IDs back to text
    /// </summary>
    public string DecodeTokens(Tensor<long> tokenIds)
    {
        var tokens = new int[tokenIds.Dimensions[1]];
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens[i] = (int)tokenIds[0, i];
        }
        return Decode(tokens);
    }
    
    /// <summary>
    /// Convert a single token ID to its text representation
    /// </summary>
    public string DecodeSingleToken(int tokenId)
    {
        if (_reverseVocab.TryGetValue(tokenId, out var token))
        {
            // Handle special tokens
            if (_specialTokens.ContainsValue(tokenId))
            {
                return token;
            }

            // Remove the "Ġ" prefix if present (used for byte-level BPE)
            if (token.StartsWith("Ġ"))
            {
                return token[1..];
            }

            return token;
        }

        return "[UNK]"; // Return unknown token marker if ID not found
    }

    private Dictionary<string, int> LoadVocabulary(string path)
    {
        var jsonString = File.ReadAllText(path);
        var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(jsonString);
        return vocab ?? throw new InvalidOperationException("Failed to load vocabulary");
    }
    
    private Dictionary<int, string> CreateReverseVocabulary(Dictionary<string, int> vocab)
    {
        return vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
    }

    private List<(string, string)> LoadMerges(string path)
    {
        var merges = new List<(string, string)>();
        var lines = File.ReadAllLines(path);
        
        // Skip first line (version info)
        for (var i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(' ');
            if (parts.Length == 2)
            {
                merges.Add((parts[0], parts[1]));
            }
        }
        
        return merges;
    }

    private Dictionary<string, int> InitializeSpecialTokens()
    {
        return new Dictionary<string, int>
        {
            {"<s>", BosTokenId},
            {"</s>", EosTokenId},
            {"<pad>", PadTokenId},
            {"<unk>", UnkTokenId}
        };
    }

    private Task<List<string>> PreProcessTextAsync(string text)
    {
        return Task.Run(() =>
        {
            // Add spaces around special tokens
            foreach (var specialToken in _specialTokens.Keys)
            {
                text = text.Replace(specialToken, $" {specialToken} ");
            }
            
            // Split on whitespace while keeping special tokens intact
            var words = new List<string>();
            var currentWord = new StringBuilder();
            var inSpecialToken = false;
            
            foreach (var c in text)
            {
                if (c == '<')
                {
                    if (currentWord.Length > 0)
                    {
                        words.Add(currentWord.ToString());
                        currentWord.Clear();
                    }
                    inSpecialToken = true;
                    currentWord.Append(c);
                }
                else if (c == '>' && inSpecialToken)
                {
                    currentWord.Append(c);
                    words.Add(currentWord.ToString());
                    currentWord.Clear();
                    inSpecialToken = false;
                }
                else if (char.IsWhiteSpace(c) && !inSpecialToken)
                {
                    if (currentWord.Length > 0)
                    {
                        words.Add(currentWord.ToString());
                        currentWord.Clear();
                    }
                }
                else
                {
                    currentWord.Append(c);
                }
            }
            
            if (currentWord.Length > 0)
            {
                words.Add(currentWord.ToString());
            }
            
            return words.Where(w => !string.IsNullOrWhiteSpace(w)).ToList();
        });
    }

    private IEnumerable<int> EncodeWord(string word)
    {
        // Check if it's a special token first
        if (_specialTokens.TryGetValue(word, out var specialTokenId))
        {
            return new[] { specialTokenId };
        }

        // Convert word to byte-level representation
        var bytePairs = GetBytePairs(word);
        
        while (true)
        {
            var bigram = GetHighestRankingPair(bytePairs);
            if (!bigram.HasValue) break;

            var (first, second) = bigram.Value;
            var newPairs = new List<string>();
            var i = 0;

            while (i < bytePairs.Count - 1)
            {
                if (i < bytePairs.Count - 1 && 
                    bytePairs[i] == first && 
                    bytePairs[i + 1] == second)
                {
                    newPairs.Add(first + second);
                    i += 2;
                }
                else
                {
                    newPairs.Add(bytePairs[i]);
                    i += 1;
                }
            }

            if (i == bytePairs.Count - 1)
            {
                newPairs.Add(bytePairs[^1]);
            }

            bytePairs = newPairs;
            if (bytePairs.Count == 1) break;
        }

        // Convert subwords to token IDs
        return bytePairs.Select(subword => 
            _vocab.TryGetValue(subword, out var id) ? id : UnkTokenId);
    }

    private List<string> GetBytePairs(string word)
    {
        // Convert word to UTF-8 bytes and create initial byte pairs
        var bytes = Encoding.UTF8.GetBytes(word);
        return bytes.Select(b => "Ġ" + b.ToString()).ToList();
    }

    private (string First, string Second)? GetHighestRankingPair(List<string> bytePairs)
    {
        var minRank = int.MaxValue;
        (string, string)? bestPair = null;

        for (var i = 0; i < bytePairs.Count - 1; i++)
        {
            var pair = (bytePairs[i], bytePairs[i + 1]);
            var rank = _bpeMerges.IndexOf(pair);
            
            if (rank != -1 && rank < minRank)
            {
                minRank = rank;
                bestPair = pair;
            }
        }

        return bestPair;
    }

    private Tensor<long> CreateTensor(List<int> tokens)
    {
        // Create tensor with batch size 1
        var tensorShape = new[] { 1, tokens.Count };
        var tensor = new DenseTensor<long>(tensorShape);
        
        for (var i = 0; i < tokens.Count; i++)
        {
            tensor[0, i] = tokens[i];
        }
        
        return tensor;
    }
}
