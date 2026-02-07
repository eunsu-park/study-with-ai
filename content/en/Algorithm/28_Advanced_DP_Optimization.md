# 29. Advanced DP Optimization

## Learning Objectives
- Understanding and implementing Convex Hull Trick (CHT)
- Divide and Conquer Optimization
- Applying Knuth Optimization
- Identifying conditions for each optimization technique
- Recognizing representative problem patterns

## 1. Overview

### When DP Optimization is Needed

```
┌─────────────────────────────────────────────────┐
│              DP Recurrence Form                  │
├─────────────────────────────────────────────────┤
│  dp[i] = min(dp[j] + cost(j, i))                │
│          j < i                                   │
│                                                  │
│  Basic: O(N²) → Optimization needed!            │
└─────────────────────────────────────────────────┘

Conditions by optimization technique:
┌────────────────┬──────────────────────────────────┐
│ CHT            │ cost = a[j] * b[i] form          │
│ D&C Opt        │ opt[i-1] ≤ opt[i] (monotonicity) │
│ Knuth Opt      │ opt[i][j-1] ≤ opt[i][j] ≤ opt[i+1][j] │
└────────────────┴──────────────────────────────────┘
```

---

## 2. Convex Hull Trick (CHT)

### Applicable Conditions

When recurrence has this form:
```
dp[i] = min/max(dp[j] + a[j] * b[i] + c[j] + d[i])
        j < i

Where:
- a[j]: depends only on j
- b[i]: depends only on i
- c[j]: depends only on j (treated as constant)
- d[i]: depends only on i (same for all j)
```

### Key Idea

For each j, define line `y = a[j] * x + (dp[j] + c[j])`, then `dp[i]` is the minimum (or maximum) of these lines at `x = b[i]`.

```
    y
    |    /
    |   /    ← Lower envelope of lines
    |  /  \
    | /    \
    |/______\_____ x
        b[i]

CHT = Manage lower (or upper) envelope of lines
```

### Basic Implementation (Sorted Case)

```python
class ConvexHullTrickMin:
    """
    CHT for minimum queries
    Conditions: slopes monotonically decreasing, query x monotonically increasing
    """
    def __init__(self):
        self.lines = []  # (slope, y-intercept)
        self.ptr = 0

    def bad(self, l1, l2, l3):
        """Check if l2 is unnecessary"""
        # If intersection(l1, l2).x >= intersection(l2, l3).x then l2 unnecessary
        return (l3[1] - l1[1]) * (l1[0] - l2[0]) <= (l2[1] - l1[1]) * (l1[0] - l3[0])

    def add_line(self, m, b):
        """Add line y = mx + b"""
        line = (m, b)
        while len(self.lines) >= 2 and self.bad(self.lines[-2], self.lines[-1], line):
            self.lines.pop()
        self.lines.append(line)

    def query(self, x):
        """Minimum value at x"""
        if not self.lines:
            return float('inf')

        # Move pointer (when x increases monotonically)
        while self.ptr < len(self.lines) - 1:
            m1, b1 = self.lines[self.ptr]
            m2, b2 = self.lines[self.ptr + 1]
            if m1 * x + b1 > m2 * x + b2:
                self.ptr += 1
            else:
                break

        m, b = self.lines[self.ptr]
        return m * x + b

# Usage: dp[i] = min(dp[j] + a[j] * b[i])
def solve_with_cht(a, b):
    n = len(a)
    dp = [0] * n
    cht = ConvexHullTrickMin()

    # Add first line (j=0)
    cht.add_line(a[0], dp[0])

    for i in range(1, n):
        dp[i] = cht.query(b[i])
        cht.add_line(a[i], dp[i])

    return dp[n-1]
```

### Li Chao Tree (General CHT)

When slopes or query order are not sorted

```python
class LiChaoTree:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.tree = {}  # Store line per node

    def add_line(self, m, b, node=1, lo=None, hi=None):
        """Add line y = mx + b"""
        if lo is None:
            lo, hi = self.lo, self.hi

        if lo > hi:
            return

        mid = (lo + hi) // 2

        if node not in self.tree:
            self.tree[node] = (m, b)
            return

        old_m, old_b = self.tree[node]

        # Compare which line is better at mid
        left_better = m * lo + b < old_m * lo + old_b
        mid_better = m * mid + b < old_m * mid + old_b

        if mid_better:
            self.tree[node] = (m, b)
            m, b = old_m, old_b

        if lo == hi:
            return

        # Recurse based on where intersection occurs
        if left_better != mid_better:
            self.add_line(m, b, 2 * node, lo, mid)
        else:
            self.add_line(m, b, 2 * node + 1, mid + 1, hi)

    def query(self, x, node=1, lo=None, hi=None):
        """Minimum value at x"""
        if lo is None:
            lo, hi = self.lo, self.hi

        if node not in self.tree:
            return float('inf')

        m, b = self.tree[node]
        result = m * x + b

        if lo == hi:
            return result

        mid = (lo + hi) // 2
        if x <= mid:
            result = min(result, self.query(x, 2 * node, lo, mid))
        else:
            result = min(result, self.query(x, 2 * node + 1, mid + 1, hi))

        return result

# Usage example
tree = LiChaoTree(-1000000, 1000000)
tree.add_line(2, 1)    # y = 2x + 1
tree.add_line(-1, 5)   # y = -x + 5
print(tree.query(0))   # 1
print(tree.query(3))   # 2 (min of 7, 2)
```

### C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

struct Line {
    ll m, b;
    ll eval(ll x) { return m * x + b; }
};

class CHT {
private:
    deque<Line> lines;

    bool bad(Line l1, Line l2, Line l3) {
        // Check if l2 is unnecessary
        return (__int128)(l3.b - l1.b) * (l1.m - l2.m) <=
               (__int128)(l2.b - l1.b) * (l1.m - l3.m);
    }

public:
    void addLine(ll m, ll b) {
        Line line = {m, b};
        while (lines.size() >= 2 && bad(lines[lines.size()-2], lines[lines.size()-1], line))
            lines.pop_back();
        lines.push_back(line);
    }

    ll query(ll x) {
        // Binary search or pointer
        int lo = 0, hi = lines.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (lines[mid].eval(x) > lines[mid + 1].eval(x))
                lo = mid + 1;
            else
                hi = mid;
        }
        return lines[lo].eval(x);
    }
};
```

---

## 3. Divide and Conquer Optimization

### Applicable Conditions

```
dp[i][j] = min(dp[i-1][k] + cost(k, j))  for k < j
           k

Condition: opt[i][j] ≤ opt[i][j+1]
           (monotonicity of optimal split point)

This holds when:
- cost function satisfies quadrangle inequality
- cost(a, c) + cost(b, d) ≤ cost(a, d) + cost(b, c)  (a ≤ b ≤ c ≤ d)
```

### Algorithm

```
1. Find optimal k for dp[i][mid]
2. dp[i][lo..mid-1] search only in k_lo..k_mid range
3. dp[i][mid+1..hi] search only in k_mid..k_hi range
4. O(N² / level) = O(N log N) per row → O(KN log N) total
```

### Implementation

```python
def dnc_optimization(n, k, cost):
    """
    Calculate dp[k][n] (partition n elements into k groups)
    cost(i, j): cost of interval [i, j)
    """
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0

    def compute(row, dp_lo, dp_hi, opt_lo, opt_hi):
        if dp_lo > dp_hi:
            return

        dp_mid = (dp_lo + dp_hi) // 2
        best_cost = INF
        best_opt = opt_lo

        for opt in range(opt_lo, min(opt_hi, dp_mid) + 1):
            current_cost = dp[row - 1][opt] + cost(opt, dp_mid)
            if current_cost < best_cost:
                best_cost = current_cost
                best_opt = opt

        dp[row][dp_mid] = best_cost

        # Divide and conquer
        compute(row, dp_lo, dp_mid - 1, opt_lo, best_opt)
        compute(row, dp_mid + 1, dp_hi, best_opt, opt_hi)

    for row in range(1, k + 1):
        compute(row, 1, n, 0, n - 1)

    return dp[k][n]

# Example: Partition array into k segments to minimize cost
def solve():
    arr = [1, 3, 2, 4, 5, 2]
    n = len(arr)
    k = 3

    # Preprocess: prefix sum for fast range sum
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    def range_sum(i, j):
        return prefix[j] - prefix[i]

    # cost(i, j) = (sum of interval [i, j))²
    def cost(i, j):
        s = range_sum(i, j)
        return s * s

    result = dnc_optimization(n, k, cost)
    print(f"Minimum cost: {result}")
```

### C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

int n, k;
vector<ll> arr, prefix;
vector<vector<ll>> dp;

ll cost(int i, int j) {
    ll sum = prefix[j] - prefix[i];
    return sum * sum;
}

void compute(int row, int lo, int hi, int opt_lo, int opt_hi) {
    if (lo > hi) return;

    int mid = (lo + hi) / 2;
    ll best = LLONG_MAX;
    int best_opt = opt_lo;

    for (int opt = opt_lo; opt <= min(opt_hi, mid - 1); opt++) {
        ll val = dp[row - 1][opt] + cost(opt, mid);
        if (val < best) {
            best = val;
            best_opt = opt;
        }
    }

    dp[row][mid] = best;

    compute(row, lo, mid - 1, opt_lo, best_opt);
    compute(row, mid + 1, hi, best_opt, opt_hi);
}

ll solve() {
    dp.assign(k + 1, vector<ll>(n + 1, LLONG_MAX));
    dp[0][0] = 0;

    for (int row = 1; row <= k; row++) {
        compute(row, 1, n, 0, n - 1);
    }

    return dp[k][n];
}
```

---

## 4. Knuth Optimization

### Applicable Conditions

```
dp[i][j] = min(dp[i][k] + dp[k][j]) + cost(i, j)
           i < k < j

Conditions:
1. cost satisfies quadrangle inequality
   cost(a, c) + cost(b, d) ≤ cost(a, d) + cost(b, c)

2. Monotonicity
   cost(b, c) ≤ cost(a, d)  when a ≤ b ≤ c ≤ d

Result: opt[i][j-1] ≤ opt[i][j] ≤ opt[i+1][j]
```

### Algorithm

```
O(N³) → O(N²)

for length in range(2, n+1):
    for i in range(n - length + 1):
        j = i + length
        # k range: opt[i][j-1] ~ opt[i+1][j]
```

### Implementation (Optimal BST)

```python
def optimal_bst(freq):
    """
    Optimal binary search tree construction cost
    freq[i]: search frequency of i-th key
    """
    n = len(freq)
    INF = float('inf')

    # prefix sum for range cost
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + freq[i]

    def cost(i, j):
        return prefix[j] - prefix[i]

    # dp[i][j]: minimum cost for interval [i, j)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    opt = [[0] * (n + 1) for _ in range(n + 1)]

    # Length 1
    for i in range(n):
        dp[i][i + 1] = freq[i]
        opt[i][i + 1] = i

    # Length 2 or more
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length
            dp[i][j] = INF

            # Knuth optimization: opt[i][j-1] ~ opt[i+1][j]
            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 <= n and j <= n else j - 1

            for k in range(lo, min(hi, j - 1) + 1):
                val = dp[i][k] + dp[k + 1][j] + cost(i, j)
                if val < dp[i][j]:
                    dp[i][j] = val
                    opt[i][j] = k

    return dp[0][n]

# Usage example
frequencies = [25, 10, 20, 5, 15, 25]
print(f"Optimal BST cost: {optimal_bst(frequencies)}")
```

### Representative Problem: Matrix Chain Multiplication

```python
def matrix_chain_multiplication(dims):
    """
    Minimum operations for matrix chain multiplication
    dims[i]: number of rows in i-th matrix (last is columns)
    """
    n = len(dims) - 1
    INF = float('inf')

    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    for i in range(n):
        opt[i][i] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            lo = opt[i][j - 1] if j > i else i
            hi = opt[i + 1][j] if i + 1 < n else j

            for k in range(lo, hi + 1):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k

    return dp[0][n - 1]

# Example: 4 matrices (10x30, 30x5, 5x60, 60x10)
print(matrix_chain_multiplication([10, 30, 5, 60, 10]))  # 4200
```

---

## 5. Condition Detection Guide

### Decision Tree

```
Recurrence: dp[i] = min(dp[j] + cost(j, i))

Is cost in form a[j] * b[i]?
├── Yes → Use CHT
│         ├── Slopes sorted? → Stack/deque CHT
│         └── Not sorted? → Li Chao Tree
└── No
    ↓
Does opt[i][j] have monotonicity?
├── Yes → D&C optimization or Knuth optimization
└── No → Regular DP (O(N²))
```

### Checking Quadrangle Inequality

```python
def check_quadrangle_inequality(cost, n):
    """
    cost(a, c) + cost(b, d) ≤ cost(a, d) + cost(b, c)
    for all a ≤ b ≤ c ≤ d
    """
    for a in range(n):
        for b in range(a, n):
            for c in range(b, n):
                for d in range(c, n):
                    lhs = cost(a, c) + cost(b, d)
                    rhs = cost(a, d) + cost(b, c)
                    if lhs > rhs:
                        return False
    return True
```

---

## 6. Time Complexity Summary

| Technique | Time Complexity | Applicable Conditions |
|-----------|----------------|----------------------|
| Basic DP | O(N²) or O(N³) | - |
| CHT (sorted) | O(N) | cost = a[j] * b[i] |
| CHT (general) | O(N log N) | cost = a[j] * b[i] |
| Li Chao Tree | O(N log N) | cost = a[j] * b[i] |
| D&C optimization | O(KN log N) | opt monotonicity |
| Knuth optimization | O(N²) | Quadrangle inequality |

---

## 7. Representative Problems

### Problem 1: Special Post Office (CHT)

```python
def special_post_office(villages, costs):
    """
    Minimize post office installation cost
    dp[i] = min(dp[j] + cost[j] * dist[i] + ...)
    """
    n = len(villages)
    cht = ConvexHullTrickMin()

    dp = [0] * (n + 1)
    cht.add_line(costs[0], dp[0])

    for i in range(1, n + 1):
        dp[i] = cht.query(villages[i - 1])
        if i < n:
            cht.add_line(costs[i], dp[i])

    return dp[n]
```

### Problem 2: Array Partitioning (D&C)

```python
def partition_array(arr, k):
    """
    Partition array into k segments to minimize sum of squares of segment sums
    """
    return dnc_optimization(len(arr), k, lambda i, j: sum(arr[i:j])**2)
```

### Problem 3: File Merging (Knuth)

```python
def merge_files(sizes):
    """
    Minimum cost to merge consecutive files
    """
    n = len(sizes)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + sizes[i]

    dp = [[0] * (n + 1) for _ in range(n + 1)]
    opt = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        opt[i][i + 1] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length
            dp[i][j] = float('inf')

            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 < n else j - 1

            for k in range(lo, hi + 1):
                val = dp[i][k] + dp[k][j] + prefix[j] - prefix[i]
                if val < dp[i][j]:
                    dp[i][j] = val
                    opt[i][j] = k

    return dp[0][n]
```

---

## 8. Common Mistakes

### Mistake 1: CHT Slope Order

```python
# Min CHT: slopes monotonically decreasing
# Max CHT: slopes monotonically increasing

# If order is wrong, use Li Chao Tree
```

### Mistake 2: Range Check

```python
# In D&C optimization
for opt in range(opt_lo, min(opt_hi, dp_mid) + 1):
    #                    ^^^ Must be less than dp_mid
```

### Mistake 3: Integer Overflow

```cpp
// cost = a[j] * b[i] requires long long
typedef long long ll;
ll cost = (ll)a[j] * b[i];
```

---

## 9. Practice Problems

| Difficulty | Problem Type | Key Concept |
|-----------|--------------|-------------|
| ★★★ | Special Forces (BOJ 4008) | CHT |
| ★★★ | Tree Cutting | CHT |
| ★★★★ | Prison Break | D&C optimization |
| ★★★★ | File Merging | Knuth optimization |
| ★★★★★ | IOI Problems | Combined |

---

## 10. Learning Roadmap

```
1. Master basic DP
   ↓
2. Understand CHT (line management)
   ↓
3. D&C optimization (using monotonicity)
   ↓
4. Knuth optimization (quadrangle inequality)
   ↓
5. Solve complex problems
```

---

## Conclusion

This completes the 30 algorithm lessons!

### Overall Lesson Summary

| Range | Topics |
|-------|--------|
| 01-05 | Basics (complexity, arrays, strings, recursion, sorting) |
| 06-10 | Core (binary search, stack/queue, trees, heaps, graphs) |
| 11-15 | Advanced (DFS/BFS, shortest paths, MST, DP, practice) |
| 16-20 | Interview essentials (hash, strings, math, topological sort, bitmask) |
| 21-25 | Advanced data structures/graphs (segment tree, trie, Fenwick, SCC, flow) |
| 26-29 | Special topics (LCA, geometry, game theory, DP optimization) |

---

## Learning Checklist

1. What recurrence form allows CHT?
2. What is opt monotonicity in D&C optimization?
3. What is the role of quadrangle inequality in Knuth optimization?
4. When is Li Chao Tree needed?
