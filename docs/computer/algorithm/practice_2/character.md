## KMP
**KMP Algorithm**

The Knuth-Morris-Pratt (KMP) algorithm is a classic string matching algorithm used to 
find the occurrence of a pattern string within a text string. It was developed by Donald Knuth,
 Vaughan Pratt, and James H. Morris in 1977. The KMP algorithm improves the efficiency of string 
 matching by avoiding unnecessary comparisons.

**Key Principles**

The KMP algorithm utilizes a preprocessing step to create a partial match table (also known
as the "next" array) that stores the lengths of the longest proper prefix which is also a 
suffix for each prefix of the pattern. This table helps in determining how far the pattern should
 be shifted when a mismatch occurs, thus avoiding redundant comparisons.

**Code Example**

Here is a Python implementation of the KMP algorithm:

```py
def compute_prefix_function(pattern):
    m = len(pattern)
    next = [0] * m
    k = 0
    for q in range(1, m):
        while k > 0 and pattern[k] != pattern[q]:
            k = next[k - 1]
        if pattern[k] == pattern[q]:
            k += 1
        next[q] = k
    return next

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    next = compute_prefix_function(pattern)
    q = 0
    for i in range(n):
        while q > 0 and pattern[q] != text[i]:
            q = next[q - 1]
        if pattern[q] == text[i]:
            q += 1
        if q == m:
            print(f"Pattern occurs at index {i - m + 1}")
            q = next[q - 1]

# Example usage
text = "ababcabcabababd"
pattern = "ababd"
kmp_search(text, pattern)
```

简单模版：
```py
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def find_occurrences(t, s):
    cur = s + "#" + t
    sz1, sz2 = len(t), len(s)
    ret = []
    lps = prefix_function(cur)
    for i in range(sz2 + 1, sz1 + sz2 + 1):
        if lps[i] == sz2:
            ret.append(i - 2 * sz2)
    return ret
```

## 前缀树

```c++
class Trie {
private:
    vector<Trie*> children;
    bool isEnd;

    Trie* searchPrefix(string prefix) {
        Trie* node = this;
        for (char ch : prefix) {
            ch -= 'a';
            if (node->children[ch] == nullptr) {
                return nullptr;
            }
            node = node->children[ch];
        }
        return node;
    }

public:
    Trie() : children(26), isEnd(false) {}

    void insert(string word) {
        Trie* node = this;
        for (char ch : word) {
            ch -= 'a';
            if (node->children[ch] == nullptr) {
                node->children[ch] = new Trie();
            }
            node = node->children[ch];
        }
        node->isEnd = true;
    }

    bool search(string word) {
        Trie* node = this->searchPrefix(word);
        return node != nullptr && node->isEnd;
    }

    bool startsWith(string prefix) {
        return this->searchPrefix(prefix) != nullptr;
    }
};
```