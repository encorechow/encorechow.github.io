---
layout: post
title: Find kth smallest elements in two sorted array
excerpt: "Preparing for interviews."
categories: [Algorithms]
tags: [code, Leetcode, two pointers, binary search]
comments: true
image:
  feature: Algorithms-In-Computer-Science.jpg
---
---

### Problem Description

> Given two sorted arrays A and B of size m and n respectively. Find the k-th smallest elements in the union of A and B. You can assume that there are no duplicates. (Subproblem of LC No.4)

*For example if A = [10, 20, 40, 60] and B =[15, 35, 50, 70, 100] and K = 4 then solution should be 35 because union of above arrays will be
C= [10,15,20,35,40,50,60,70,100] and fourth smallest element is 35.*

### Brute Force (Trivial Solution)

The trivial solution for this problem can be observed from above example. Merge two arrays to a single sorted array the k-th element can be accessed directly. Example shows C is merged from A and B and 35 is 4th element in C. If reader knows the merge sort algorithm, O(m+n) space complexity should be concluded (length of C). Likewise, time complexity is O(m+n) as well.

### Slightly Better Solution

Derived from trivial solution, we can apply **two pointers** idea in merge sort. But instead of actually merging two arrays into a single sorted array, we merely traverse both sorted arrays k steps in order to find kth smallest element. Specifically, the pointer that has smaller value of two should increment one step forward. After k steps traversal, the kth smallest element will be found. This solution is extra-space-free with total time complexity O(k) (k steps traversal)

### Awesome Solution among All

We know that n and m can be extremely large, so as the k. Linear complexity is still not ideal. Since the two arrays are sorted, we should somehow take advantage of this to obtain a logarithmic complexity. When we come up with logarithmic complexity, the first thing we should be able to consider is binary search.

Basically, binary search achieves logarithmic complexity by separating input space into two halves in each iteration. The next iteration we are supposed to throw away one half of the array then do the same things on the other half. But how do we exactly apply binary search in two sorted arrays with different size? That can be a tricky part.

Consider an initial guess that how the numbers that smaller than kth element distributed in two arrays. If they are evenly distributed in two arrays, that is, k/2 elements that smaller or equal than kth element in array *A* and the rest of k - k/2 elements in array *B* (Why k - k/2 instead of k/2? Consider k is an odd number). It is simple to conclude that the larger one between *A[k/2]* and *B[k/2]* is the kth element. But in the case that they are not evenly distributed, that does not work out.

Let's say the k/2 index in array A is *i* and the k - k/2 index in array B is *j*. In general, there are **three** cases (here I use 0 based indexing):

+ **A[i-1] > B[j-1]**: The portion of *B[0:j-1]* (*j-1* inclusive) can be discarded, since this portion is strictly smaller than *A[i-1]* The other portion we can discard is *A[i:len(A)]* (*i* inclusive). The reason is that *A[i-1]* is definitely larger than *len(B[0:j-1]) + len(A[0:i-2])* elements and there is no duplicates. If kth element locates in *A[i:len(A)]*, we will get a contradiction that kth smallest element in the position that larger than k elements and that's apparently not true. Figure below shows how it works:  

![Figure1]({{ site.url }}/assets/kth_1.jpg)  

+ **A[i-1] < B[j-1]**: This case is the reverse case of above. Figure below shows how it works:  

![Figure2]({{ site.url }}/assets/kth_2.jpg)  

+ **Base case**:
   1. Consider if k equals to 1. The smallest number between A[k-1] and B[k-1] should be returned (0 based indexing).  
   2. As long as either of two arrays has been completely discarded, we can just return kth smallest element in the other array.  

From now on, we are all set for coding.

```python
def kth_smallest(A, B, k):
    if k == 0 or (len(A) == 0 and len(B) == 0) or (k > len(A) + len(B)):
        print("Invalid input.")
        return None
    if k == 1:
        return min(A[0], B[0])
    if len(A) == 0 or len(B) == 0:
        return A[k-1] if len(B) == 0 else B[k-1]

    i = min(k // 2, len(A))
    j = k - i

    if A[i-1] > B[j-1]:
        return kth_smallest(A[:i], B[j:], k-j)
    else:
        return kth_smallest(A[i:], B[:j], k-i)
    return None
```

### Time Complexity Analysis

The total time complexity has something to do with k in this solution at first glance. Suppose we have array A with size N and array B with size M, we are guaranteed to cut something from A and B more or less. Thus thinking under the entire input space M + N, the final time complexity should be O(log(M + N)) and the base of logarithmic doesn't matter.
