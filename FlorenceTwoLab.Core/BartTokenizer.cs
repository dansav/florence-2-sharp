using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using FlorenceTwoLab.Core.Utils;

namespace FlorenceTwoLab.Core;

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
    private readonly Trie _specialTokens;
    private readonly Dictionary<string, string> _cache;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<int, byte> _byteDecoder;
    private readonly Dictionary<byte, int> _byteEncoder;

    private readonly Regex _pattern;

    private BartTokenizer(
        Dictionary<string, int> encoder,
        Dictionary<int, string> decoder,
        Dictionary<(string, string), int> bpeRanks,
        IReadOnlyCollection<string>? addedTokens = null)
    {
        // Initialize collections
        _encoder = encoder;
        _decoder = decoder;
        _bpeRanks = bpeRanks;
        _cache = new Dictionary<string, string>();
        _specialTokens = new Trie();
        foreach (var specialToken in Enumerable.Concat([PadToken, BosToken, EosToken, UnkToken, MaskToken], addedTokens ?? []))
        {
            _specialTokens.Add(specialToken);
        }

        // Initialize byte encoder/decoder
        (_byteEncoder, _byteDecoder) = InitializeByteMappings();

        // Initialize regex pattern
        _pattern = MyRegex();
    }

    public int VocabSize => _encoder.Count;

    public static async Task<BartTokenizer> FromPretrainedAsync(Stream mergesStream, Stream vocabStream, Stream? addedTokensStream = null)
    {
        // Load vocabulary
        var (encoder, decoder) = await LoadVocabularyAsync(vocabStream);

        // Load merges
        var bpeRanks = await LoadMergesAsync(mergesStream);

        // Load added tokens if provided
        IReadOnlyCollection<string>? addedTokens = null;
        if (addedTokensStream is not null)
        {
            addedTokens = await LoadAddedTokensAsync(addedTokensStream, encoder, decoder);
        }

        return new BartTokenizer(encoder, decoder, bpeRanks, addedTokens);
    }

    public static async Task<BartTokenizer> FromPretrainedAsync(string metaDataDirectory)
    {
        var vocabPath = Path.Combine(metaDataDirectory, BaseVocabFileName);
        var addedTokensPath = Path.Combine(metaDataDirectory, AdditionalVocabFileName);
        var mergesPath = Path.Combine(metaDataDirectory, MergesFileName);

        if (!File.Exists(vocabPath)) throw new ArgumentException("Vocabulary file not found", nameof(vocabPath));
        if (!File.Exists(mergesPath)) throw new ArgumentException("Merges file not found", nameof(mergesPath));

        await using var vocabStream = File.OpenRead(vocabPath);
        await using var mergesStream = File.OpenRead(mergesPath);
        await using var addedTokensStream = File.Exists(addedTokensPath) ? File.OpenRead(addedTokensPath) : null;

        return await FromPretrainedAsync(mergesStream, vocabStream, addedTokensStream);
    }

    private static async Task<(Dictionary<string, int> encoder, Dictionary<int, string> decoder)> LoadVocabularyAsync(Stream vocabStream)
    {
        return await Task.Run(() =>
        {
            var encoder = new Dictionary<string, int>();
            var decoder = new Dictionary<int, string>();

            var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabStream);
            if (vocab is null) throw new ArgumentException("Vocabulary file is empty or invalid", nameof(vocabStream));

            foreach (var kvp in vocab)
            {
                encoder[kvp.Key] = kvp.Value;
                decoder[kvp.Value] = kvp.Key;
            }

            return (encoder, decoder);
        });
    }

    private static async Task<Dictionary<(string, string), int>> LoadMergesAsync(Stream mergesStream)
    {
        return await Task.Run(async () =>
        {
            using var reader = new StreamReader(mergesStream);
            var bpeRanks = new Dictionary<(string, string), int>();

            int i = 0;
            bool skip = true;
            while (await reader.ReadLineAsync(CancellationToken.None).ConfigureAwait(false) is { } merge)
            {
                if (skip) // Skip header line
                {
                    skip = false;
                    continue;
                }

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

    private static async Task<IReadOnlyCollection<string>> LoadAddedTokensAsync(
        Stream addedTokensStream,
        Dictionary<string, int> encoder,
        Dictionary<int, string> decoder)
    {
        return await Task.Run(() =>
        {
            var addedTokens = JsonSerializer.Deserialize<Dictionary<string, int>>(addedTokensStream);
            if (addedTokens is null) throw new ArgumentException("Added tokens file is empty or invalid", nameof(addedTokensStream));

            var added = new List<string>();
            foreach (var kvp in addedTokens)
            {
                added.Add(kvp.Key);
                encoder[kvp.Key] = kvp.Value;
                decoder[kvp.Value] = kvp.Key;
            }

            return added;
        });
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
        var parts = _specialTokens.Split(text);
        var result = new List<string>();

        foreach (var part in parts)
        {
            if (_specialTokens.Contains(part))
            {
                result.Add(part);
                continue;
            }

            foreach (var match in _pattern.Matches(part).Select(m => m.Value))
            {
                var encodedToken = EncodeToken(match);
                result.AddRange(BytePairEncode(encodedToken).Split(' '));
            }
        }

        return result;
    }

    private string EncodeToken(string token)
    {
        if (string.IsNullOrEmpty(token)) return token;

        var encoded = new StringBuilder();
        var bytes = Encoding.UTF8.GetBytes(token);

        foreach (var b in bytes)
        {
            if (_byteEncoder.TryGetValue(b, out var value))
            {
                encoded.Append((char)value);
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
        if (_cache.TryGetValue(token, out var encode))
            return encode;

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
        var unknown = _encoder[UnkToken];
        var ids = new List<int>();
        foreach (var token in tokens)
        {
            ids.Add(_encoder.GetValueOrDefault(token, unknown));
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

            if (_decoder.TryGetValue(id, out var token))
            {
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
            if (_byteDecoder.TryGetValue(c, out var value))
            {
                bytes.Add(value);
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }


    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")]
    private static partial Regex MyRegex();
}
