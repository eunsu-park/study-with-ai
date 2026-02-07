# Dynamic Programming

## Overview

Dynamic Programming (DP) is an algorithm design technique that solves complex problems by breaking them down into simpler subproblems. It improves efficiency by storing results of overlapping subproblems.

---

## Table of Contents

1. [DP Concepts](#1-dp-concepts)
2. [Memoization vs Tabulation](#2-memoization-vs-tabulation)
3. [Basic DP Problems](#3-basic-dp-problems)
4. [1D DP](#4-1d-dp)
5. [2D DP](#5-2d-dp)
6. [String DP](#6-string-dp)
7. [Practice Problems](#7-practice-problems)

---

## 1. DP Concepts

### DP Conditions

```
Conditions for applying Dynamic Programming:

1. Optimal Substructure
   - Optimal solution consists of optimal solutions to subproblems
   - Example: Subpaths of shortest paths are also shortest paths

2. Overlapping Subproblems
   - Same subproblems are solved multiple times
   - Example: fib(3) calculated multiple times in Fibonacci
```

### DP vs Divide and Conquer

```
┌────────────────┬─────────────────┬─────────────────┐
│                │ DP              │ Divide&Conquer  │
├────────────────┼─────────────────┼─────────────────┤
│ Subproblem     │ Overlapping     │ Independent     │
│ overlap        │ (store)         │                 │
│ Computation    │ Store & reuse   │ Independent calc│
│ Examples       │ Fibonacci, LCS  │ Merge/Quick sort│
└────────────────┴─────────────────┴─────────────────┘
```

### Understanding Through Fibonacci

```
Fibonacci: fib(n) = fib(n-1) + fib(n-2)

Regular recursion (redundant calculations):
                  fib(5)
                 /      \
            fib(4)      fib(3)
           /    \       /    \
       fib(3)  fib(2) fib(2) fib(1)
       /   \
   fib(2) fib(1)

→ fib(3) twice, fib(2) three times
→ Time complexity: O(2^n)

DP (store and reuse):
fib(1)=1 → fib(2)=1 → fib(3)=2 → fib(4)=3 → fib(5)=5

→ Each value calculated once
→ Time complexity: O(n)
```

---

## 2. Memoization vs Tabulation

### Memoization (Top-Down)

```
Top-down: Recursion + Caching

Characteristics:
- Computes only needed parts (Lazy Evaluation)
- Uses recursion (watch for stack overflow)
- Intuitive recurrence implementation
```

```cpp
// C++ - Memoization
int memo[100];

int fib(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];

    memo[n] = fib(n-1) + fib(n-2);
    return memo[n];
}
```

```python
# Python - Memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

# Or using dictionary
def fib_memo(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

### Tabulation (Bottom-Up)

```
Bottom-up: Iteration + Table

Characteristics:
- Computes all subproblems
- Uses loops (no stack overflow)
- Space optimization possible
```

```c
// C - Tabulation
int fib(int n) {
    if (n <= 1) return n;

    int dp[n + 1];
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }

    return dp[n];
}

// Space optimized: O(1)
int fibOptimized(int n) {
    if (n <= 1) return n;

    int prev2 = 0, prev1 = 1;

    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}
```

```python
# Python - Tabulation
def fib(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# Space optimized
def fib_optimized(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1

    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2

    return prev1
```

---

## 3. Basic DP Problems

### 3.1 Climbing Stairs

```
Problem: n stairs, can climb 1 or 2 steps at a time
        How many ways to reach step n?

Recurrence: dp[i] = dp[i-1] + dp[i-2]
- dp[i-1]: climb 1 step from position i-1
- dp[i-2]: climb 2 steps from position i-2

Example: n=4
dp[1]=1, dp[2]=2, dp[3]=3, dp[4]=5
```

```python
def climb_stairs(n):
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2

    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]
```

### 3.2 Coin Change (Minimum Count)

```
Problem: Given coin denominations, find minimum coins to make amount

Coins: [1, 3, 4], Amount: 6

dp[i] = minimum coins to make amount i

dp[0]=0
dp[1]=min(dp[0]+1)=1                    (one 1-coin)
dp[2]=min(dp[1]+1)=2                    (two 1-coins)
dp[3]=min(dp[2]+1, dp[0]+1)=1           (one 3-coin)
dp[4]=min(dp[3]+1, dp[1]+1, dp[0]+1)=1  (one 4-coin)
dp[5]=min(dp[4]+1, dp[2]+1, dp[1]+1)=2  (4+1 or 3+1+1)
dp[6]=min(dp[5]+1, dp[3]+1, dp[2]+1)=2  (3+3)
```

```cpp
// C++
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);  // Initialize with impossible value
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }

    return dp[amount] > amount ? -1 : dp[amount];
}
```

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

### 3.3 Coin Change (Number of Ways)

```
Problem: Given coin types, count ways to make amount

Coins: [1, 2, 5], Amount: 5

Recurrence: dp[i] += dp[i - coin]
Note: To avoid counting order, iterate by coin type

Initial dp: [1, 0, 0, 0, 0, 0]

coin=1: [1, 1, 1, 1, 1, 1]
coin=2: [1, 1, 2, 2, 3, 3]
coin=5: [1, 1, 2, 2, 3, 4]

Answer: 4 ways (1+1+1+1+1, 1+1+1+2, 1+2+2, 5)
```

```python
def coin_combinations(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]
```

---

## 4. 1D DP

### 4.1 Maximum Subarray Sum (Kadane's Algorithm)

```
Problem: Find maximum sum of contiguous subarray

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]

dp[i] = max sum of subarray ending at position i
dp[i] = max(arr[i], dp[i-1] + arr[i])

dp: [-2, 1, -2, 4, 3, 5, 6, 1, 5]

Maximum: 6 (subarray: [4, -1, 2, 1])
```

```cpp
// C++
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];

    for (int i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]);
        maxSum = max(maxSum, currentSum);
    }

    return maxSum;
}
```

```python
def max_sub_array(nums):
    max_sum = current_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum
```

### 4.2 House Robber

```
Problem: Cannot rob adjacent houses, find maximum amount

Houses: [2, 7, 9, 3, 1]

dp[i] = maximum amount considering houses up to i
dp[i] = max(dp[i-1], dp[i-2] + arr[i])
- dp[i-1]: don't rob house i
- dp[i-2] + arr[i]: rob house i

dp[0]=2
dp[1]=max(2, 7)=7
dp[2]=max(7, 2+9)=11
dp[3]=max(11, 7+3)=11
dp[4]=max(11, 11+1)=12

Answer: 12 (2 + 9 + 1)
```

```cpp
// C++
int rob(vector<int>& nums) {
    if (nums.empty()) return 0;
    if (nums.size() == 1) return nums[0];

    int prev2 = 0, prev1 = 0;

    for (int num : nums) {
        int curr = max(prev1, prev2 + num);
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}
```

```python
def rob(nums):
    if not nums:
        return 0

    prev2, prev1 = 0, 0

    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)

    return prev1
```

### 4.3 Longest Increasing Subsequence (LIS)

```
Problem: Find length of longest increasing subsequence

Array: [10, 9, 2, 5, 3, 7, 101, 18]

dp[i] = length of LIS ending at position i
dp[i] = max(dp[j] + 1) for all j < i where arr[j] < arr[i]

dp: [1, 1, 1, 2, 2, 3, 4, 4]

Answer: 4 (e.g., [2, 3, 7, 101] or [2, 5, 7, 101])
```

```cpp
// C++ - O(n²)
int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}

// C++ - O(n log n) Binary search
int lengthOfLISFast(vector<int>& nums) {
    vector<int> tails;

    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }

    return tails.size();
}
```

```python
import bisect

def length_of_lis(nums):
    # O(n log n)
    tails = []

    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)
```

---

## 5. 2D DP

### 5.1 0/1 Knapsack Problem

```
Problem: Knapsack capacity W, given weights and values of items
        Find maximum value that fits in knapsack

Items: [(weight, value)] = [(2,3), (3,4), (4,5), (5,6)]
Capacity: W = 5

dp[i][w] = max value with first i items and capacity w

        w=0  1   2   3   4   5
i=0      0   0   3   3   3   3   (item1: weight2, value3)
i=1      0   0   3   4   4   7   (item2: weight3, value4)
i=2      0   0   3   4   5   7   (item3: weight4, value5)
i=3      0   0   3   4   5   7   (item4: weight5, value6)

Answer: 7 (item1 + item2: weight 2+3=5, value 3+4=7)
```

```cpp
// C++
int knapsack(int W, vector<int>& weights, vector<int>& values) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            // Don't include item i-1
            dp[i][w] = dp[i-1][w];

            // Include item i-1
            if (weights[i-1] <= w) {
                dp[i][w] = max(dp[i][w],
                               dp[i-1][w - weights[i-1]] + values[i-1]);
            }
        }
    }

    return dp[n][W];
}

// Space optimized: O(W)
int knapsackOptimized(int W, vector<int>& weights, vector<int>& values) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {  // Reverse order!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    return dp[W];
}
```

```python
def knapsack(W, weights, values):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][W]
```

### 5.2 Grid Paths

```
Problem: In m×n grid, count paths from top-left to bottom-right
        Can only move right or down

Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1]

3×3 grid:
1 1 1
1 2 3
1 3 6

Answer: 6
```

```cpp
// C++
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1));

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }

    return dp[m-1][n-1];
}
```

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]
```

### 5.3 Minimum Path Sum

```
Problem: Given grid with costs in each cell, find minimum cost path

Grid:
1 3 1
1 5 1
4 2 1

dp[i][j] = minimum cost to reach (i,j)
dp:
1  4  5
2  7  6
6  8  7

Answer: 7 (1→3→1→1→1)
```

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    dp[0][0] = grid[0][0]

    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[m-1][n-1]
```

---

## 6. String DP

### 6.1 Longest Common Subsequence (LCS)

```
Problem: Find length of longest common subsequence of two strings

s1 = "ABCDGH"
s2 = "AEDFHR"

dp[i][j] = LCS length of s1[0..i-1] and s2[0..j-1]

    ""  A  E  D  F  H  R
""   0  0  0  0  0  0  0
A    0  1  1  1  1  1  1
B    0  1  1  1  1  1  1
C    0  1  1  1  1  1  1
D    0  1  1  2  2  2  2
G    0  1  1  2  2  2  2
H    0  1  1  2  2  3  3

Answer: 3 (ADH)
```

```cpp
// C++
int longestCommonSubsequence(string s1, string s2) {
    int m = s1.length(), n = s2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}
```

```python
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

### 6.2 Edit Distance

```
Problem: Minimum operations to transform one string to another
        Operations: insert, delete, replace

s1 = "horse"
s2 = "ros"

dp[i][j] = minimum operations to transform s1[0..i-1] to s2[0..j-1]

    ""  r  o  s
""   0  1  2  3
h    1  1  2  3
o    2  2  1  2
r    3  2  2  2
s    4  3  3  2
e    5  4  4  3

Answer: 3 (horse → rorse → rose → ros)
```

```cpp
// C++
int minDistance(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({dp[i-1][j],      // Delete
                                    dp[i][j-1],      // Insert
                                    dp[i-1][j-1]});  // Replace
            }
        }
    }

    return dp[m][n];
}
```

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # Delete
                                   dp[i][j-1],      # Insert
                                   dp[i-1][j-1])    # Replace

    return dp[m][n]
```

---

## 7. Practice Problems

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) | LeetCode | Basic |
| ⭐ | [Fibonacci Function](https://www.acmicpc.net/problem/1003) | BOJ | Basic |
| ⭐⭐ | [Coin Change](https://leetcode.com/problems/coin-change/) | LeetCode | Coins |
| ⭐⭐ | [LCS](https://www.acmicpc.net/problem/9251) | BOJ | String |
| ⭐⭐ | [0/1 Knapsack](https://www.acmicpc.net/problem/12865) | BOJ | Knapsack |
| ⭐⭐⭐ | [LIS](https://www.acmicpc.net/problem/11053) | BOJ | LIS |
| ⭐⭐⭐ | [Edit Distance](https://leetcode.com/problems/edit-distance/) | LeetCode | String |

---

## DP Problem Approach

```
1. Define state: What does dp[i] represent?
2. Establish recurrence: Relationship between dp[i] and previous states
3. Set initial values: Base cases
4. Determine calculation order: Based on dependencies
5. Extract answer: Where is the final answer located?
```

---

## Next Steps

- [19_Greedy_Algorithms.md](./19_Greedy_Algorithms.md) - Greedy Algorithms

---

## References

- [DP Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns)
- [VisuAlgo - DP](https://visualgo.net/en/recursion)
- Introduction to Algorithms (CLRS) - Chapter 15
