---
layout: post
title: Word Break
excerpt: "Algorithm coding practice. (Online judge passed)"
categories: [Algorithms]
tags: [code, Leetcode, algorithm, dynamic programming]
comments: true
---

### Problem Description
> Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

*For example,  
Given  s = "leetcode",  
dict = ["leet", "code"].  
Return true because "leetcode" can be segmented as "leet code".*

### Brute Force (Trivial Solution)
Naively, This problem can be solved by comparing the words in dictionary with the substring that has same length with each word. By recursively do so, the algorithm will return true if the string can be split by the dictionary, false otherwise.

### Dynamic Programming (Coding by python)

Subproblem defined as: the substring s(0, i) can be segmented by the dictionary if s(0, i-j) for j = len(word in dictionary) can be segmented by the dictionary as well.

Recurrence relation:  

- Define an array dp[i] = true if s(0, i-1) can be segmented by dictionary, false otherwise.
- dp[i] = (dp[i-j]) & (s(i-j, i) == word(j)) for j = len(word in dictionary)
- dp[0] = true as base case.

```python  

from collections import defaultdict
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: bool
        """
        dp = defaultdict(lambda: False)
        dp[0] = True

        for i in range(len(s)):
            if not dp[i]:
                continue
            for word in wordDict:
                forward = i + len(word)

                if dp[forward]:
                    continue
                dp[forward] = (s[i:forward] == word) & (dp[i]);

        return dp[len(s)]
```

###### Time Complexity
Suppose n is the size of string and m is the size of dictionary, above solution has time complexity $$O(n\times m)$$;


### Simpler and More Efficient (Coding by java)

If the dictionary is extremely large, above dp solution is not efficent enough. Since the dictionary is a set, the time complexity of **contains** operation for set just O(1).  

Unlike above solution that exhaustively go through the dictionary, this algorithm maintain a flags array. flags[j] = true if and only if the dictionary contains substring s(i,j) & flag[i-1] = true **(the substring before the character that will be checked should also be able to segmented)** for 0 <= j < len(s).  


```java
public class Solution {
    public boolean wordBreak(String s, Set<String> wordDict) {
        if (s == null || s.length() == 0 || wordDict == null ||wordDict.size() == 0){
            return false;
        }
        int slen = s.length();
        boolean[] flags = new boolean[slen];

        for (int i = 0; i < slen; i++){
            boolean prev = true;
            if (i > 0){
                prev = flags[i-1];
            }
            if (prev){
                for (int j = i; j < slen; j++){
                    String word = s.substring(i,j+1);

                    if (wordDict.contains(word)){
                        flags[j] = true;    
                    }
                }
            }
        }
        return flags[slen-1];
    }
}
```


###### Time Complexity  
Suppose n is the size of string and m is the size of dictionary, above solution has time complexity $$O(n^2)$$;
