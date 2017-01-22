---
layout: post
title: Longest Substring Without Repeating Characters
excerpt: "Algorithm coding practice. (Online judge passed)"
categories: [Algorithms]
tags: [code, Leetcode, algorithm, string, hash table]
comments: true
---

### Problem Description
> Given a string, find the length of the longest substring without repeating characters.

*Example:  
Given "abcabcbb", the answer is "abc", which the length is 3.  
Given "bbbbb", the answer is "b", with the length of 1.  
Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.*


### Sliding Window

Key points of algorithm:  
- use unordered_map to store if character has appeared;  
- if character has appeared twice, set the characters before the first occurrence of repeated character to false (not appeared);  
- update maximum length and move the pointer forward.  

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {

        // Two Pointers
        int p1 = 0, p2 = 0;

        int maxLen = 0;

        unordered_map<char, bool> umap;

        while (p1 < s.length() && p2 < s.length()){
            // If s[p1] hasn't appeared before
            if (!umap[s[p1]]){
                umap[s[p1]] = true;
                maxLen = max(maxLen, p1-p2+1);
                p1++;
            // s[p1] has appeared twice
            }else{
                // clear the occurrence state of characters that appeared before the first occurrence of repeated character.
                umap[s[p2]] = false;
                maxLen = max(maxLen, p1-p2);
                p2++;
            }
        }
        return maxLen;

    }
};
```

### Sliding Window Optimized

Optimization: Store the index of character into the map so that we can skip the characters which appeared before the first occurrence of duplicates directly.

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {

        int maxLen = 0;

        unordered_map<char, int> umap;

        for (int p1 = 0, p2 = 0; p1 < s.length(); p1++){

            if (umap.find(s[p1]) != umap.end()){
                p2 = max(umap[s[p1]], p2);
            }
            maxLen = max(maxLen, p1 - p2 + 1);
            umap[s[p1]] = p1 + 1;
        }

        return maxLen;

    }
};
```
