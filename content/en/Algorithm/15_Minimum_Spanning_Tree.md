# Minimum Spanning Tree

## Overview

A Minimum Spanning Tree (MST) is a tree that connects all vertices of a graph while minimizing the sum of edge weights. We'll learn about Kruskal and Prim algorithms, as well as the Union-Find data structure.

---

## Table of Contents

1. [MST Concept](#1-mst-concept)
2. [Union-Find](#2-union-find)
3. [Kruskal Algorithm](#3-kruskal-algorithm)
4. [Prim Algorithm](#4-prim-algorithm)
5. [Algorithm Comparison](#5-algorithm-comparison)
6. [Practice Problems](#6-practice-problems)

---

## 1. MST Concept

### Spanning Tree

```
Spanning Tree: A subgraph that includes all vertices
               with no cycles

Conditions:
- Vertices: V
- Edges: V-1
- All vertices connected
- No cycles
```

### Minimum Spanning Tree (MST)

```
MST: A spanning tree with minimum sum of edge weights

    (1)──4──(2)
    │╲      │╲
    2  1    5  3
    │    ╲  │    ╲
   (3)──6──(4)──7──(5)

MST (total weight: 11):
    (1)──4──(2)
     ╲       ╲
      1       3
        ╲      ╲
        (4)──────(5)
         │
         2(connected to 3, not shown properly in diagram)

Actual MST:
(1)-1-(4), (1)-2-(3), (2)-4-(1), (2)-3-(5)
→ 1+2+4+3 = 10? Or other combination
```

### MST Properties

```
1. Cut Property: When dividing a graph into two sets,
   the minimum weight edge crossing the cut is in the MST

2. Cycle Property: The maximum weight edge in a cycle
   is not in the MST

3. Uniqueness: If all edge weights are distinct, the MST is unique
```

---

## 2. Union-Find (Disjoint Set Union)

### Concept

```
Disjoint Sets: Sets with no common elements

Operations:
- Find(x): Returns the representative element of the set containing x
- Union(x, y): Merges the sets containing x and y

Use Cases:
- Cycle detection
- Connected component management
- Kruskal's algorithm
```

### Basic Implementation

```c
// C
#define MAX_N 100001

int parent[MAX_N];

void init(int n) {
    for (int i = 0; i < n; i++) {
        parent[i] = i;  // Each element is its own parent
    }
}

int find(int x) {
    if (parent[x] == x) {
        return x;
    }
    return find(parent[x]);
}

void unite(int x, int y) {
    int px = find(x);
    int py = find(y);
    if (px != py) {
        parent[px] = py;
    }
}
```

### Optimization 1: Path Compression

```
Connect all nodes on the path directly to the root during Find

     (5)              (5)
      │               /|\
     (3)      →     (1)(2)(3)
     /│              │
   (1)(2)            (4)
    │
   (4)

Time Complexity: Nearly O(1) (Amortized)
```

```c
// C - Path Compression
int find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);  // Recursively connect to root
    }
    return parent[x];
}
```

### Optimization 2: Union by Rank

```
Attach smaller tree to larger tree

  Tree1 (Rank 2)    Tree2 (Rank 1)
       (a)              (b)
      / │ \              │
    (c)(d)(e)           (f)

After union:
       (a)
      /│╲  \
    (c)(d)(e)(b)
              │
             (f)
```

```c
// C - Path Compression + Union by Rank
int parent[MAX_N];
int rank_arr[MAX_N];

void init(int n) {
    for (int i = 0; i < n; i++) {
        parent[i] = i;
        rank_arr[i] = 0;
    }
}

int find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

void unite(int x, int y) {
    int px = find(x);
    int py = find(y);

    if (px == py) return;

    // Attach smaller rank tree to larger rank tree
    if (rank_arr[px] < rank_arr[py]) {
        parent[px] = py;
    } else if (rank_arr[px] > rank_arr[py]) {
        parent[py] = px;
    } else {
        parent[py] = px;
        rank_arr[px]++;
    }
}
```

### C++/Python Implementation

```cpp
// C++
class UnionFind {
private:
    vector<int> parent, rank_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;

        return true;
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

```python
# Python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

---

## 3. Kruskal Algorithm

### Concept

```
Sort edges by weight and select edges that don't create cycles

Principle:
1. Sort all edges in ascending order by weight
2. Select from the smallest edge
3. Skip if it creates a cycle (check with Union-Find)
4. Stop when V-1 edges are selected

Time Complexity: O(E log E)
```

### Example

```
Graph:
   (0)──7──(1)
    │╲    ╱│
    5  8 9  7
    │    ╲╱ │
   (2)──5──(3)

Sorted edges: (2,3,5), (0,2,5), (0,1,7), (1,3,7), (0,3,8), (1,2,9)

Selection process:
1. (2,3,5) selected → No cycle ✓
2. (0,2,5) selected → No cycle ✓
3. (0,1,7) selected → No cycle ✓
4. V-1=3 edges selected, complete

MST: (2,3), (0,2), (0,1)
Total weight: 5+5+7 = 17
```

### Implementation

```c
// C
#define MAX_E 100001

typedef struct {
    int u, v, weight;
} Edge;

Edge edges[MAX_E];
int parent[MAX_E];

int cmp(const void* a, const void* b) {
    return ((Edge*)a)->weight - ((Edge*)b)->weight;
}

int find(int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]);
    return parent[x];
}

int kruskal(int V, int E) {
    // Initialize
    for (int i = 0; i < V; i++)
        parent[i] = i;

    // Sort
    qsort(edges, E, sizeof(Edge), cmp);

    int mstWeight = 0;
    int edgeCount = 0;

    for (int i = 0; i < E && edgeCount < V - 1; i++) {
        int pu = find(edges[i].u);
        int pv = find(edges[i].v);

        if (pu != pv) {
            parent[pu] = pv;
            mstWeight += edges[i].weight;
            edgeCount++;
        }
    }

    return mstWeight;
}
```

```cpp
// C++
struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

int kruskal(int V, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());

    UnionFind uf(V);
    int mstWeight = 0;
    int edgeCount = 0;

    for (const auto& e : edges) {
        if (edgeCount >= V - 1) break;

        if (uf.unite(e.u, e.v)) {
            mstWeight += e.weight;
            edgeCount++;
        }
    }

    return mstWeight;
}
```

```python
# Python
def kruskal(V, edges):
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(V)

    mst_weight = 0
    edge_count = 0

    for u, v, w in edges:
        if edge_count >= V - 1:
            break

        if uf.union(u, v):
            mst_weight += w
            edge_count += 1

    return mst_weight
```

---

## 4. Prim Algorithm

### Concept

```
Gradually expand the MST starting from a starting vertex

Principle:
1. Start from an arbitrary vertex
2. Among edges going out from vertices in the MST,
   select the edge with the smallest weight
3. Add the new vertex to the MST
4. Stop when all vertices are included

Time Complexity:
- Priority Queue: O(E log V)
- Adjacency Matrix: O(V²)
```

### Example

```
Graph (starting from 0):
   (0)──7──(1)
    │╲    ╱│
    5  8 9  7
    │    ╲╱ │
   (2)──5──(3)

Steps:
1. Start: MST = {0}
   Adjacent edges: (0,1,7), (0,2,5), (0,3,8)
   Select: (0,2,5) → MST = {0,2}

2. Adjacent edges: (0,1,7), (0,3,8), (2,3,5)
   Select: (2,3,5) → MST = {0,2,3}

3. Adjacent edges: (0,1,7), (3,1,7)
   Select: (0,1,7) or (3,1,7) → MST = {0,1,2,3}

Result: Total weight = 5+5+7 = 17
```

### Implementation (Priority Queue)

```cpp
// C++
int prim(int V, const vector<vector<pair<int,int>>>& adj) {
    vector<bool> inMST(V, false);
    // {weight, vertex}
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;

    int mstWeight = 0;
    pq.push({0, 0});  // Starting vertex

    while (!pq.empty()) {
        auto [w, u] = pq.top();
        pq.pop();

        if (inMST[u]) continue;

        inMST[u] = true;
        mstWeight += w;

        for (auto [v, weight] : adj[u]) {
            if (!inMST[v]) {
                pq.push({weight, v});
            }
        }
    }

    return mstWeight;
}
```

```python
# Python
import heapq

def prim(V, adj):
    in_mst = [False] * V
    pq = [(0, 0)]  # (weight, vertex)
    mst_weight = 0

    while pq:
        w, u = heapq.heappop(pq)

        if in_mst[u]:
            continue

        in_mst[u] = True
        mst_weight += w

        for v, weight in adj[u]:
            if not in_mst[v]:
                heapq.heappush(pq, (weight, v))

    return mst_weight
```

### Implementation (Adjacency Matrix, V²)

```cpp
// C++ - Better for dense graphs
int primMatrix(int V, const vector<vector<int>>& adj) {
    vector<int> key(V, INT_MAX);
    vector<bool> inMST(V, false);

    key[0] = 0;
    int mstWeight = 0;

    for (int count = 0; count < V; count++) {
        // Select vertex with minimum key value
        int u = -1;
        for (int v = 0; v < V; v++) {
            if (!inMST[v] && (u == -1 || key[v] < key[u])) {
                u = v;
            }
        }

        inMST[u] = true;
        mstWeight += key[u];

        // Update key values of adjacent vertices
        for (int v = 0; v < V; v++) {
            if (adj[u][v] && !inMST[v] && adj[u][v] < key[v]) {
                key[v] = adj[u][v];
            }
        }
    }

    return mstWeight;
}
```

---

## 5. Algorithm Comparison

### Kruskal vs Prim

```
┌─────────────┬──────────────────┬──────────────────┐
│             │ Kruskal          │ Prim             │
├─────────────┼──────────────────┼──────────────────┤
│ Approach    │ Edge-centric     │ Vertex-centric   │
│ Data Struct │ Union-Find       │ Priority Queue   │
│ Time        │ O(E log E)       │ O(E log V)       │
│ Best for    │ Sparse graphs    │ Dense graphs     │
│ Complexity  │ Relatively simple│ Relatively complex│
└─────────────┴──────────────────┴──────────────────┘
```

### Selection Criteria

```
Sparse graphs (E ≈ V): Kruskal is better
Dense graphs (E ≈ V²): Prim is better

Edge list input: Kruskal is better
Adjacency list input: Prim is better
```

---

## 6. Practice Problems

### Problem 1: Minimum Spanning Tree

Find the total weight of the MST for the given graph.

<details>
<summary>Solution Code</summary>

```python
def solution(V, edges):
    # Kruskal
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(V)

    total = 0
    count = 0

    for u, v, w in edges:
        if count >= V - 1:
            break
        if uf.union(u, v):
            total += w
            count += 1

    return total
```

</details>

### Problem 2: City Division Plan

Divide N villages into 2 groups and connect each group with minimum cost.

<details>
<summary>Hint</summary>

After constructing the MST, remove the largest edge to create 2 groups

</details>

<details>
<summary>Solution Code</summary>

```python
def divide_villages(V, edges):
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(V)

    mst_edges = []

    for u, v, w in edges:
        if uf.union(u, v):
            mst_edges.append(w)
            if len(mst_edges) == V - 1:
                break

    # Remove the largest edge
    return sum(mst_edges) - max(mst_edges)
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Algorithm |
|--------|------|--------|----------|
| ⭐⭐ | [Minimum Spanning Tree](https://www.acmicpc.net/problem/1197) | BOJ | Kruskal/Prim |
| ⭐⭐ | [Sanggeun's Travel](https://www.acmicpc.net/problem/9372) | BOJ | MST Concept |
| ⭐⭐⭐ | [City Division Plan](https://www.acmicpc.net/problem/1647) | BOJ | MST Application |
| ⭐⭐⭐ | [Network Connection](https://www.acmicpc.net/problem/1922) | BOJ | MST |
| ⭐⭐⭐ | [Min Cost to Connect](https://leetcode.com/problems/min-cost-to-connect-all-points/) | LeetCode | Prim |

---

## Template Summary

### Union-Find

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### Kruskal

```python
def kruskal(V, edges):
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(V)
    total = 0
    for u, v, w in edges:
        if uf.union(u, v):
            total += w
    return total
```

### Prim

```python
def prim(V, adj):
    in_mst = [False] * V
    pq = [(0, 0)]
    total = 0
    while pq:
        w, u = heapq.heappop(pq)
        if in_mst[u]:
            continue
        in_mst[u] = True
        total += w
        for v, weight in adj[u]:
            if not in_mst[v]:
                heapq.heappush(pq, (weight, v))
    return total
```

---

## Next Steps

- [16_LCA_and_Tree_Queries.md](./16_LCA_and_Tree_Queries.md) - LCA, Tree Queries

---

## References

- [MST Visualization](https://visualgo.net/en/mst)
- [Union-Find Tutorial](https://cp-algorithms.com/data_structures/disjoint_set_union.html)
- Introduction to Algorithms (CLRS) - Chapter 23
