---
layout: post
title: Easy Dynamic Programming Problems
excerpt: "Algorithm coding practice. (Online judge passed)"
categories: [Algorithms]
tags: [code, Leetcode, algorithm, dynamic programming]
comments: true
---

## 1. Climbing Stairs

> You are climbing a stair case. It takes n steps to reach to the top.  
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?


### Formula

base case: dp[1] = 1, dp[2] = 2

dp[i] = dp[i-1] + dp[i-2]

### Memorization

Recursive version of dynamic programming using extra container to store computed sub-problems to avoid repeated computation.

#### C++

```c++
class Solution {
public:
    int climbStairs(int n) {
        map<int, int> mem;
        return helper(n, mem);
    }
    int helper(int n, map<int, int>& mem){
        if (n == 1 || n == 2){
            mem[n] = n == 1 ? 1 : 2;
            return mem[n];
        }
        if (mem.find(n) != mem.end()){
            return mem[n];
        }
        mem[n] = helper(n-2, mem) + helper(n-1, mem);
        return mem[n];
    }
};
```

### Iteration

Iteratively solve the sub-problems

#### Python

```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """

        dp = {}
        dp[1] = 1
        dp[2] = 2

        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

```


## 2. House Robber

> You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.  
Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.


### Formula

base case: dp[-1] = 0, dp[0] = nums[0]

dp[i] = max(dp[i-1], dp[i-2] + nums[i])

### Memorization

##### C++
```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        map<int, int> mem;
        return helper(nums, mem, nums.size()-1);
    }

    int helper(vector<int>& nums, map<int, int>& mem, int idx){
        if (idx == -1 || idx == 0){
            return (idx == -1 ? 0 : nums[0]);
        }
        if (mem.find(idx) != mem.end()){
            return mem[idx];
        }

        mem[idx] = max(helper(nums, mem, idx-1), helper(nums, mem, idx-2) + nums[idx]);
        return mem[idx];
    }
};
```


### Iteration


##### Python
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        dp = {}
        dp[0] = 0
        dp[1] = nums[0]

        for i in range(2, len(nums)+1):
            dp[i] = max(dp[i-2] + nums[i-1], dp[i-1])

        return dp[len(nums)]

```


## 3. Best Time to Buy and Sell Stock

> Say you have an array for which the ith element is the price of a given stock on day i.
If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.


### Formula

base case: dp[0] = 0

dp[i] = max(dp[i-1], prices[i] - min_so_far)


### Memorization

Memorization will cause stack overflow error for large input array.

##### c++
```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() <= 0){
            return 0;
        }
        map<int, int> mem;
        return helper(prices, mem, prices.size()-1, prices[prices.size()-1]);
    }
    int helper(vector<int>& prices, map<int, int>& mem, int idx, int max_val){
        if (idx == 0){
            return 0;
        }

        int max_next = prices[idx] < max_val ? max_val : prices[idx];
        mem[idx] = max(helper(prices, mem, idx-1, max_next), max_val - prices[idx]);
        return mem[idx];
    }
};
```


### Iteration

##### Python
```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        dp = {}
        dp[-1], dp[0] = 0, 0
        min_val = prices[0] if len(prices) > 0 else 0

        for i in range(1, len(prices)):
            min_val = prices[i] if prices[i] < min_val else min_val
            dp[i] = max(dp[i-1], prices[i] - min_val)

        return dp[len(prices)-1]
```
