---
layout: post
title: Reverse Words in A String
excerpt: "Algorithm coding practice. (Online judge passed)"
categories: [Algorithms]
tags: [code, Leetcode, algorithm, string]
comments: true
---

### Problem Description
> Given an input string, reverse the string word by word.

*For example:  
Given s = "the sky is blue",  
return "blue is sky the".*


### With Auxiliary Array

The idea is to iterate from the end of the string, once encountered a **space** we append the word into the result with a **space** except the last word in the string.  

#### C++

```c++
class Solution {
public:
    void reverseWords(string &s) {
        trim(s);

        string res;
        int wordEnd = s.length();
        for (int i = s.length()-1; i >= 0; i--){
            if (s[i] == ' '){
                wordEnd = i;
            }else if (i == 0 || s[i-1] == ' '){
                if (!res.empty()){
                    res.append(" ");
                }
                res += s.substr(i, wordEnd-i);
            }
        }
        s = res;
    }
    void trim(string &s){
        s.erase(0, s.find_first_not_of(" "));
        s.erase(s.find_last_not_of(" ")+1);
    }
};
```

#### Python

```python
class Solution(object):
  def reverseWords(self, s):
    """
    :type s: str
    :rtype: str
    """

    s = s.strip()
    word_end = len(s)
    res = ''
    for i in range(len(s)-1, -1, -1):
        if s[i] == ' ':
            word_end = i
        elif i == 0 or s[i-1] == ' ':
            res += ' ' + s[i:word_end]
    return res.strip()
```


### In Place Solution

This can be done by:  
  - Reverse the individual word in the initial string;  
  - Overwrite redundant space by moving the characters back;  
  - Reverse the entire string at once;  
  - Eliminate the rest of useless contents;  

#### C++

```c++
void reverseWords(string &s) {
        int end = 0;
        for (int i = 0; i < s.length(); i++){
            if (s[i] != ' '){

                // Append a space
                if (end != 0){
                    s[end++] = ' ';
                }
                int j = i;

                // Copy from initial string
                while(j < s.length() && s[j] != ' '){
                    s[end++] = s[j++];
                }

                // Reverse a word
                swap(s, end-(j-i), end-1);
                i = j;

            }
        }
        // Reverse the string
        swap(s, 0, end-1);
        // Erase rest of the contents
        s.erase(end);

    }
    void swap(string &s, int i1, int i2){
        for (; i1 < i2; i1++, i2--){
            int temp = s[i1];
            s[i1] = s[i2];
            s[i2] = temp;
        }
    }
```
