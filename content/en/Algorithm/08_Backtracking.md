# Backtracking

## Overview

Backtracking is a technique for finding solutions by exploring possibilities and returning when hitting a dead end. It reduces unnecessary exploration through pruning.

---

## Table of Contents

1. [Backtracking Concept](#1-backtracking-concept)
2. [Permutations and Combinations](#2-permutations-and-combinations)
3. [N-Queens](#3-n-queens)
4. [Subsets](#4-subsets)
5. [Sudoku](#5-sudoku)
6. [Practice Problems](#6-practice-problems)

---

## 1. Backtracking Concept

### Basic Principles

```
Backtracking:
1. Build solution incrementally
2. Go back to previous step when condition not met
3. Reduce search space through pruning

DFS + Condition checking + Backtracking
```

### State Space Tree

```
Permutation search when N=3:

                    []
         /          |          \
       [1]         [2]         [3]
       / \         / \         / \
    [1,2][1,3] [2,1][2,3] [3,1][3,2]
      |    |     |    |     |    |
   [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]

Pruning example: Skip entire subtree when first element violates condition
```

### Basic Template

```python
def backtrack(candidate):
    if is_solution(candidate):
        output(candidate)
        return

    for next_choice in choices(candidate):
        if is_valid(next_choice):  # Pruning
            candidate.append(next_choice)
            backtrack(candidate)
            candidate.pop()  # Undo
```

---

## 2. Permutations and Combinations

### 2.1 Permutation

```
Arrange r items from n with order
nPr = n! / (n-r)!

All permutations of [1, 2, 3]:
[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
```

```cpp
// C++
void permute(vector<int>& nums, int start, vector<vector<int>>& result) {
    if (start == nums.size()) {
        result.push_back(nums);
        return;
    }

    for (int i = start; i < nums.size(); i++) {
        swap(nums[start], nums[i]);
        permute(nums, start + 1, result);
        swap(nums[start], nums[i]);  // Undo
    }
}

vector<vector<int>> permutations(vector<int>& nums) {
    vector<vector<int>> result;
    permute(nums, 0, result);
    return result;
}
```

```python
def permutations(nums):
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result

# Or use itertools
from itertools import permutations
list(permutations([1, 2, 3]))
```

### 2.2 Combination

```
Choose r items from n without order
nCr = n! / (r! × (n-r)!)

Choose 2 from [1, 2, 3, 4]:
[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]
```

```cpp
// C++
void combine(int n, int r, int start, vector<int>& current,
             vector<vector<int>>& result) {
    if (current.size() == r) {
        result.push_back(current);
        return;
    }

    for (int i = start; i <= n; i++) {
        current.push_back(i);
        combine(n, r, i + 1, current, result);
        current.pop_back();
    }
}

vector<vector<int>> combinations(int n, int r) {
    vector<vector<int>> result;
    vector<int> current;
    combine(n, r, 1, current, result);
    return result;
}
```

```python
def combinations(n, r):
    result = []

    def backtrack(start, current):
        if len(current) == r:
            result.append(current[:])
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result

# Or use itertools
from itertools import combinations
list(combinations([1, 2, 3, 4], 2))
```

### 2.3 Permutations/Combinations with Repetition

```python
# Permutation with repetition: Can choose same element multiple times
def permutations_with_repetition(nums, r):
    result = []

    def backtrack(current):
        if len(current) == r:
            result.append(current[:])
            return

        for num in nums:
            current.append(num)
            backtrack(current)
            current.pop()

    backtrack([])
    return result

# Combination with repetition
def combinations_with_repetition(nums, r):
    result = []

    def backtrack(start, current):
        if len(current) == r:
            result.append(current[:])
            return

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i, current)  # i not i+1
            current.pop()

    backtrack(0, [])
    return result
```

---

## 3. N-Queens

### Problem

```
Place N queens on N×N chessboard so they cannot attack each other

Queen attack range: horizontal, vertical, diagonal

4×4 example (one solution):
. Q . .
. . . Q
Q . . .
. . Q .
```

### Algorithm

```
Place queens row by row:
1. Try placing queen in first row
2. Place queen in next row (check conflicts)
3. Backtrack if conflict
4. Output solution when N queens placed

Conflict check:
- Same column: cols[col] == True
- Diagonal1 (↘): row - col value is same
- Diagonal2 (↙): row + col value is same
```

### Implementation

```cpp
// C++
class NQueens {
private:
    int n;
    vector<bool> cols, diag1, diag2;
    vector<vector<string>> results;

    void backtrack(int row, vector<int>& queens) {
        if (row == n) {
            results.push_back(generateBoard(queens));
            return;
        }

        for (int col = 0; col < n; col++) {
            if (cols[col] || diag1[row - col + n - 1] || diag2[row + col])
                continue;

            queens[row] = col;
            cols[col] = diag1[row - col + n - 1] = diag2[row + col] = true;

            backtrack(row + 1, queens);

            cols[col] = diag1[row - col + n - 1] = diag2[row + col] = false;
        }
    }

    vector<string> generateBoard(const vector<int>& queens) {
        vector<string> board(n, string(n, '.'));
        for (int i = 0; i < n; i++) {
            board[i][queens[i]] = 'Q';
        }
        return board;
    }

public:
    vector<vector<string>> solveNQueens(int n) {
        this->n = n;
        cols.assign(n, false);
        diag1.assign(2 * n - 1, false);
        diag2.assign(2 * n - 1, false);

        vector<int> queens(n);
        backtrack(0, queens);

        return results;
    }
};
```

```python
def solve_n_queens(n):
    results = []
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row, queens):
        if row == n:
            board = ['.' * q + 'Q' + '.' * (n - q - 1) for q in queens]
            results.append(board)
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1, queens + [col])

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0, [])
    return results

# Count solutions only
def count_n_queens(n):
    count = 0
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count
```

---

## 4. Subsets

### Generate All Subsets

```
Subsets of [1, 2, 3]:
[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]

Total: 2^n subsets
```

```cpp
// C++
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;

    function<void(int)> backtrack = [&](int start) {
        result.push_back(current);

        for (int i = start; i < nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(i + 1);
            current.pop_back();
        }
    };

    backtrack(0);
    return result;
}

// Bitmask approach
vector<vector<int>> subsetsBitmask(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> result;

    for (int mask = 0; mask < (1 << n); mask++) {
        vector<int> subset;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                subset.push_back(nums[i]);
            }
        }
        result.push_back(subset);
    }

    return result;
}
```

```python
def subsets(nums):
    result = []

    def backtrack(start, current):
        result.append(current[:])

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result

# Bitmask
def subsets_bitmask(nums):
    n = len(nums)
    result = []

    for mask in range(1 << n):
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        result.append(subset)

    return result
```

### Subsets with Target Sum

```python
def subset_sum(nums, target):
    result = []

    def backtrack(start, current, current_sum):
        if current_sum == target:
            result.append(current[:])
            return

        if current_sum > target:  # Pruning
            return

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current, current_sum + nums[i])
            current.pop()

    backtrack(0, [], 0)
    return result
```

---

## 5. Sudoku

### Problem

```
9×9 grid, each row/column/3×3 box has 1-9 exactly once

5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
. 9 8 | . . . | . 6 .
------+-------+------
8 . . | . 6 . | . . 3
4 . . | 8 . 3 | . . 1
7 . . | . 2 . | . . 6
------+-------+------
. 6 . | . . . | 2 8 .
. . . | 4 1 9 | . . 5
. . . | . 8 . | . 7 9
```

### Implementation

```python
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False

        # Check column
        for r in range(9):
            if board[r][col] == num:
                return False

        # Check 3×3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False

        return True

    def solve():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(board, row, col, num):
                            board[row][col] = num

                            if solve():
                                return True

                            board[row][col] = '.'  # Backtrack

                    return False  # All numbers failed

        return True  # No empty cell = completed

    solve()
```

---

## 6. Practice Problems

### Problem 1: All Permutations of String

Generate all permutations without duplicates when string has duplicate characters

<details>
<summary>Solution Code</summary>

```python
def permute_unique(nums):
    result = []
    nums.sort()

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return

        for i in range(len(remaining)):
            # Skip duplicates
            if i > 0 and remaining[i] == remaining[i-1]:
                continue

            backtrack(current + [remaining[i]],
                     remaining[:i] + remaining[i+1:])

    backtrack([], nums)
    return result
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐⭐ | [N and M](https://www.acmicpc.net/problem/15649) | Baekjoon | Permutation |
| ⭐⭐ | [N-Queens](https://www.acmicpc.net/problem/9663) | Baekjoon | N-Queens |
| ⭐⭐ | [Subsets](https://leetcode.com/problems/subsets/) | LeetCode | Subsets |
| ⭐⭐⭐ | [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) | LeetCode | Sudoku |
| ⭐⭐⭐ | [Combination Sum](https://leetcode.com/problems/combination-sum/) | LeetCode | Combination |

---

## Backtracking Template

```python
def backtrack(state):
    if is_goal(state):
        save_solution(state)
        return

    for choice in get_choices(state):
        if is_valid(choice, state):
            make_choice(state, choice)
            backtrack(state)
            undo_choice(state, choice)
```

---

## Next Steps

- [09_Trees_and_BST.md](./09_Trees_and_BST.md) - Trees, BST

---

## References

- [Backtracking](https://www.geeksforgeeks.org/backtracking-algorithms/)
- Introduction to Algorithms (CLRS) - Backtracking
