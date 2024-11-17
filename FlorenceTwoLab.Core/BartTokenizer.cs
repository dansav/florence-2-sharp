using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace FlorenceTwoLab.Core;

public record class AddedToken(
    string Content,
    bool Lstrip = false,
    bool Rstrip = false,
    bool SingleWord = false,
    bool Normalized = true,
    bool Special = false);

public partial class BartTokenizer
{
    public const string BaseVocabFileName = "vocab.json";
    public const string AdditionalVocabFileName = "added_tokens.json";
    public const string MergesFileName = "merges.txt";

    public const string PadToken = "<pad>";
    public const string BosToken = "<s>";
    public const string EosToken = "</s>";
    public const string UnkToken = "<unk>";
    public const string MaskToken = "<mask>";

    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<int, string> _decoder;
    private readonly HashSet<string> _specialTokens;
    private readonly Dictionary<string, string> _cache;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<int, byte> _byteDecoder;
    private readonly Dictionary<byte, int> _byteEncoder;

    private readonly Regex _pattern;

    private BartTokenizer(
        Dictionary<string, int> encoder,
        Dictionary<int, string> decoder,
        Dictionary<(string, string), int> bpeRanks,
        IReadOnlyCollection<AddedToken>? addedTokens = null)
    {
        // Initialize collections
        _encoder = encoder;
        _decoder = decoder;
        _bpeRanks = bpeRanks;
        _cache = new Dictionary<string, string>();
        _specialTokens = new HashSet<string> { PadToken, BosToken, EosToken, UnkToken, MaskToken };

        AddedTokens = addedTokens;

        // Initialize byte encoder/decoder
        (_byteEncoder, _byteDecoder) = InitializeByteMappings();

        // Initialize regex pattern
        _pattern = MyRegex();
    }

    public int VocabSize => _encoder.Count;

    public IReadOnlyCollection<AddedToken>? AddedTokens { get; private set; }

    public static async Task<BartTokenizer> FromPretrainedAsync(string metaDataDirectory)
    {
        var vocabPath = Path.Combine(metaDataDirectory, BaseVocabFileName);
        var addedTokensPath = Path.Combine(metaDataDirectory, AdditionalVocabFileName);
        var mergesPath = Path.Combine(metaDataDirectory, MergesFileName);

        // Load vocabulary
        var (encoder, decoder) = await LoadVocabularyAsync(vocabPath);

        // Load merges
        var bpeRanks = await LoadMergesAsync(mergesPath);

        // Load added tokens if provided
        IReadOnlyCollection<AddedToken>? addedTokens = null;
        if (!string.IsNullOrEmpty(addedTokensPath))
        {
            addedTokens = await LoadAddedTokensAsync(addedTokensPath, encoder, decoder);
        }

        return new BartTokenizer(encoder, decoder, bpeRanks, addedTokens);
    }

    private static async Task<(Dictionary<string, int> encoder, Dictionary<int, string> decoder)> LoadVocabularyAsync(
        string vocabPath)
    {
        var json = await File.ReadAllTextAsync(vocabPath);
        return await Task.Run(() =>
        {
            var encoder = new Dictionary<string, int>();
            var decoder = new Dictionary<int, string>();

            var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(json);
            if (vocab is null) throw new ArgumentException("Vocabulary file is empty or invalid", nameof(vocabPath));

            foreach (var kvp in vocab)
            {
                encoder[kvp.Key] = kvp.Value;
                decoder[kvp.Value] = kvp.Key;
            }

            return (encoder, decoder);
        });
    }

    private static async Task<Dictionary<(string, string), int>> LoadMergesAsync(string path)
    {
        var merges = (await File.ReadAllLinesAsync(path)).Skip(1); // Skip version header

        return await Task.Run(() =>
        {
            var bpeRanks = new Dictionary<(string, string), int>();

            int i = 0;
            foreach (var merge in merges)
            {
                var parts = merge.Split();
                if (parts.Length == 2)
                {
                    bpeRanks[(parts[0], parts[1])] = i;
                    i++;
                }
            }

            return bpeRanks;
        });
    }

    private static async Task<IReadOnlyCollection<AddedToken>> LoadAddedTokensAsync(
        string path,
        Dictionary<string, int> encoder,
        Dictionary<int, string> decoder)
    {
        var json = await File.ReadAllTextAsync(path);
        var addedTokens = JsonSerializer.Deserialize<Dictionary<string, int>>(json);
        if (addedTokens is null) throw new ArgumentException("Added tokens file is empty or invalid", nameof(path));

        var added = new List<AddedToken>();
        foreach (var kvp in addedTokens)
        {
            added.Add(new AddedToken(kvp.Key, Special: true));
            encoder[kvp.Key] = kvp.Value;
            decoder[kvp.Value] = kvp.Key;
        }

        return added;
    }

    private static (Dictionary<byte, int> ByteEncoder, Dictionary<int, byte> ByteDecoder) InitializeByteMappings()
    {
        var byteEncoder = new Dictionary<byte, int>();
        var byteDecoder = new Dictionary<int, byte>();

        // Initialize basic byte mappings
        var bytes = new List<int>();
        bytes.AddRange(Enumerable.Range('!', '~' - '!' + 1));
        bytes.AddRange(Enumerable.Range('¡', '¬' - '¡' + 1));
        bytes.AddRange(Enumerable.Range('®', 'ÿ' - '®' + 1));

        var chars = new List<int>(bytes);
        int n = 0;

        for (int b = 0; b < 256; b++)
        {
            if (!bytes.Contains(b))
            {
                bytes.Add(b);
                chars.Add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < bytes.Count; i++)
        {
            byteEncoder[(byte)bytes[i]] = chars[i];
            byteDecoder[chars[i]] = (byte)bytes[i];
        }

        return (byteEncoder, byteDecoder);
    }

    public List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        foreach (Match match in _pattern.Matches(text))
        {
            var token = match.Value;
            var encodedToken = EncodeToken(token);
            tokens.AddRange(BytePairEncode(encodedToken).Split(' '));
        }

        return tokens;
    }

    private string EncodeToken(string token)
    {
        if (string.IsNullOrEmpty(token)) return token;

        var encoded = new StringBuilder();
        var bytes = Encoding.UTF8.GetBytes(token);

        foreach (var b in bytes)
        {
            if (_byteEncoder.ContainsKey(b))
            {
                encoded.Append((char)_byteEncoder[b]);
            }
        }

        return encoded.ToString();
    }

    private HashSet<(string, string)> GetPairs(List<string> word)
    {
        var pairs = new HashSet<(string, string)>();
        string prevChar = word[0];
        for (int i = 1; i < word.Count; i++)
        {
            pairs.Add((prevChar, word[i]));
            prevChar = word[i];
        }

        return pairs;
    }

    private string BytePairEncode(string token)
    {
        if (_cache.ContainsKey(token))
            return _cache[token];

        var word = token.Select(c => c.ToString()).ToList();
        if (word.Count <= 1)
            return token;

        while (true)
        {
            var pairs = GetPairs(word);
            if (pairs.Count == 0) break;

            var minRank = int.MaxValue;
            (string, string)? bigram = null;

            foreach (var pair in pairs)
            {
                if (_bpeRanks.TryGetValue(pair, out int rank))
                {
                    if (rank < minRank)
                    {
                        minRank = rank;
                        bigram = pair;
                    }
                }
            }

            if (!bigram.HasValue) break;

            var (first, second) = bigram.Value;
            var newWord = new List<string>();
            var i = 0;

            while (i < word.Count)
            {
                var j = word.IndexOf(first, i);
                if (j == -1)
                {
                    newWord.AddRange(word.Skip(i));
                    break;
                }

                newWord.AddRange(word.Skip(i).Take(j - i));
                i = j;

                if (word[i] == first && i < word.Count - 1 && word[i + 1] == second)
                {
                    newWord.Add(first + second);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i += 1;
                }
            }

            word = newWord;
            if (word.Count == 1) break;
        }

        var result = string.Join(" ", word);
        _cache[token] = result;
        return result;
    }

    public List<int> Encode(string text, bool convertToLowerCase = false)
    {
        if (convertToLowerCase)
        {
            text = text.ToLower();
        }

        var tokens = Tokenize(text);
        return ConvertTokensToIds(tokens);
    }

    public List<int> ConvertTokensToIds(List<string> tokens)
    {
        var ids = new List<int>();
        foreach (var token in tokens)
        {
            if (_encoder.ContainsKey(token))
            {
                ids.Add(_encoder[token]);
            }
            else
            {
                ids.Add(_encoder[UnkToken]);
            }
        }

        return ids;
    }

    public string Decode(List<int> ids, bool skipSpecialTokens = false)
    {
        var tokens = new List<string>();
        foreach (var id in ids.SkipWhile(i => i < 3)) // Skip initial special tokens
        {
            // We could decode a special object instead of a string to optimize the use of location tokens
            // if (id > 50264)
            // {
            //     if (id >= 50269 && id <= 51268)
            //     {
            //         Debug.WriteLine($"special location token for '{id - 50269}' ({id})");
            //     }
            //     else
            //     {
            //         Debug.WriteLine($"special token {id} => '{_decoder.GetValueOrDefault(id, "unknown")}'");                    
            //     }
            // }
            
            if (_decoder.ContainsKey(id))
            {
                var token = _decoder[id];
                if (!skipSpecialTokens || !_specialTokens.Contains(token))
                {
                    tokens.Add(token);
                }
            }
        }

        var text = string.Join(" ", tokens);
        var bytes = new List<byte>();

        foreach (char c in text)
        {
            if (_byteDecoder.ContainsKey(c))
            {
                bytes.Add(_byteDecoder[c]);
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }


    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")]
    private static partial Regex MyRegex();
}
