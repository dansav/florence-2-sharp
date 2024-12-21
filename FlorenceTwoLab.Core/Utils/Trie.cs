namespace FlorenceTwoLab.Core.Utils;

public class Trie
{
    private readonly TrieNode _root = new();
    private readonly HashSet<string> _tokens = new();

    public void Add(string token)
    {
        if (string.IsNullOrEmpty(token))
            return;

        _tokens.Add(token);
        var current = _root;

        foreach (char c in token)
        {
            if (!current.Children.ContainsKey(c))
            {
                current.Children[c] = new TrieNode();
            }
            current = current.Children[c];
        }

        current.IsEndOfToken = true;
    }

    public List<string> Split(string text)
    {
        var result = new List<string>();
        if (string.IsNullOrEmpty(text))
            return result;

        // Dictionary to keep track of active matches
        // Key is the starting position, Value is the node we're at in the trie
        var states = new Dictionary<int, TrieNode>();
        
        // List of split points in the text
        var offsets = new List<int> { 0 };
        
        for (int current = 0; current < text.Length; current++)
        {
            char currentChar = text[current];
            
            // Process existing states
            var statesToRemove = new List<int>();
            var completeMatch = false;
            int matchStart = -1;
            int matchEnd = -1;

            foreach (var state in states)
            {
                if (state.Value.Children.TryGetValue(currentChar, out var nextNode))
                {
                    states[state.Key] = nextNode;
                    if (nextNode.IsEndOfToken)
                    {
                        // Found a complete match
                        completeMatch = true;
                        matchStart = state.Key;
                        matchEnd = current + 1;
                        break;
                    }
                }
                else
                {
                    statesToRemove.Add(state.Key);
                }
            }

            // Remove states that couldn't be extended
            foreach (var key in statesToRemove)
            {
                states.Remove(key);
            }

            // Try to start new match from current position
            if (_root.Children.ContainsKey(currentChar))
            {
                var node = _root.Children[currentChar];
                states[current] = node;
                // Check if single-character token
                if (node.IsEndOfToken)
                {
                    completeMatch = true;
                    matchStart = current;
                    matchEnd = current + 1;
                }
            }

            // Handle complete match
            if (completeMatch)
            {
                offsets.Add(matchStart);
                offsets.Add(matchEnd);
                states.Clear();
                current = matchEnd - 1; // -1 because loop will increment
            }
        }

        // Add final offset if not already present
        if (offsets.Last() != text.Length)
            offsets.Add(text.Length);

        // Convert offsets to substrings
        for (int i = 0; i < offsets.Count - 1; i++)
        {
            int start = offsets[i];
            int length = offsets[i + 1] - start;
            if (length > 0)
            {
                result.Add(text.Substring(start, length));
            }
        }

        return result;
    }

    public bool Contains(string token) => _tokens.Contains(token);
}

public class TrieNode
{
    public Dictionary<char, TrieNode> Children { get; } = new();
    
    public bool IsEndOfToken { get; set; }
}
