---
layout: post
title: Depth-First Search Problem Set
excerpt: "Algorithm coding practice. (Online judge passed)"
categories: [Algorithms]
tags: [code, Leetcode, algorithm, dfs, backtracking]
comments: true
---

## 1. Number of Islands

> Given a 2d grid map of <span style="color: red">'1's</span> (land) and <span style="color: red">'0's</span> (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.


Example 1:  

```markdown  
11110  
11010  
11000  
00000  
```
Answer: 1  

Example 2:

```markdown  
11000  
11000  
00100  
00011  
```
Answer: 3  


### DFS

Relatively easier solution to recursively erase the "island rock" by changing '1' to '0';

#### C++
```c++
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.size() == 0 || grid[0].size() == 0){
            return 0;
        }

        int count = 0;

        for (int i = 0; i < grid.size(); i++){
            for (int j = 0; j < grid[0].size(); j++){
                if (grid[i][j] == '1'){
                    count++;
                    erase_island(grid, i, j);
                }
            }
        }
        return count;
    }

    void erase_island(vector<vector<char>>& grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == '0'){
            return;
        }

        grid[i][j] = '0';

        erase_island(grid, i+1, j);
        erase_island(grid, i-1, j);
        erase_island(grid, i, j+1);
        erase_island(grid, i, j-1);
    }
};
```


### BFS

#### Python

Uses queue data structure to push the processed surroundings and pop to deal with next surroudings;

```python
def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0
        cnt = 0
        queue = deque()

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (grid[i][j] == '1'):
                    cnt += 1
                    queue.appendleft((i, j))

                    self.check_surrounding(grid, queue, i, j)
        return cnt
        
    def check_surrounding(self, grid, queue, i, j):
        while len(queue) != 0:
            i, j = queue.pop()
            grid[i][j] = '0'

            if i > 0 and grid[i-1][j] == '1':
                queue.appendleft((i - 1, j))
                grid[i-1][j] = '0'  
            if i < len(grid)-1 and grid[i+1][j] == '1':
                queue.appendleft((i + 1, j))
                grid[i+1][j] = '0'
            if j > 0 and grid[i][j-1] == '1':
                queue.appendleft((i, j - 1))
                grid[i][j-1] = '0'
            if j < len(grid[0])-1 and grid[i][j+1] == '1':
                queue.appendleft((i, j + 1))
                grid[i][j+1] = '0'

```
