# Trees and Binary Search Trees (Tree and BST)

## Overview

A tree is a non-linear data structure that represents hierarchical structures. Binary Search Trees (BST) support efficient search, insertion, and deletion operations.

---

## Table of Contents

1. [Tree Basic Concepts](#1-tree-basic-concepts)
2. [Tree Traversal](#2-tree-traversal)
3. [Binary Search Tree](#3-binary-search-tree)
4. [BST Operations](#4-bst-operations)
5. [Balanced Tree Concepts](#5-balanced-tree-concepts)
6. [Practice Problems](#6-practice-problems)

---

## 1. Tree Basic Concepts

### Terminology

```
        (A) ← Root
       / | \
     (B)(C)(D) ← Internal Nodes
     / \     \
   (E)(F)    (G) ← Leaf Nodes

- Root: Topmost node (A)
- Leaf: Nodes with no children (E, F, C, G)
- Edge: Connection between nodes
- Parent/Child: A is B's parent, B is A's child
- Sibling: Nodes with the same parent (B, C, D)
- Depth: Distance from root (A=0, B=1, E=2)
- Height: Distance to the deepest leaf
```

### Binary Tree

```
Each node has at most 2 children

      (1)
     /   \
   (2)   (3)
   / \   /
 (4)(5)(6)

Special Binary Trees:
- Complete Binary Tree: All levels filled except last
- Full Binary Tree: All levels completely filled
- Skewed Tree: Children only on one side
```

### Node Structure

```c
// C
typedef struct Node {
    int data;
    struct Node* left;
    struct Node* right;
} Node;

Node* createNode(int data) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    node->left = node->right = NULL;
    return node;
}
```

```cpp
// C++
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
```

```python
# Python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
```

---

## 2. Tree Traversal

### Traversal Methods

```
      (1)
     /   \
   (2)   (3)
   / \
 (4)(5)

Preorder: Root → Left → Right
  1 → 2 → 4 → 5 → 3

Inorder: Left → Root → Right
  4 → 2 → 5 → 1 → 3

Postorder: Left → Right → Root
  4 → 5 → 2 → 3 → 1

Level-order: Level by level, left to right
  1 → 2 → 3 → 4 → 5
```

### Recursive Implementation

```cpp
// C++
void preorder(TreeNode* root) {
    if (!root) return;
    cout << root->val << " ";  // Visit
    preorder(root->left);
    preorder(root->right);
}

void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    cout << root->val << " ";  // Visit
    inorder(root->right);
}

void postorder(TreeNode* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    cout << root->val << " ";  // Visit
}
```

```python
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

### Iterative Implementation (Stack)

```cpp
// C++ - Inorder Traversal
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;

    while (curr || !st.empty()) {
        while (curr) {
            st.push(curr);
            curr = curr->left;
        }

        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }

    return result;
}
```

```python
def inorder_iterative(root):
    result = []
    stack = []
    curr = root

    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left

        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right

    return result
```

### Level-order Traversal (BFS)

```cpp
// C++
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();
        vector<int> level;

        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);

            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        result.push_back(level);
    }

    return result;
}
```

```python
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

---

## 3. Binary Search Tree (BST)

### BST Property

```
For every node:
- All values in left subtree < node value
- All values in right subtree > node value

       (8)
      /   \
    (3)   (10)
    / \      \
  (1)(6)    (14)
     / \    /
   (4)(7)(13)

Inorder traversal: 1, 3, 4, 6, 7, 8, 10, 13, 14 (sorted!)
```

### BST Operation Complexity

```
┌──────────┬─────────────┬─────────────┐
│ Operation│ Average     │ Worst       │
├──────────┼─────────────┼─────────────┤
│ Search   │ O(log n)    │ O(n)        │
│ Insert   │ O(log n)    │ O(n)        │
│ Delete   │ O(log n)    │ O(n)        │
└──────────┴─────────────┴─────────────┘

Worst case: Skewed tree (becomes like a linked list)
```

---

## 4. BST Operations

### 4.1 Search

```cpp
// C++
TreeNode* search(TreeNode* root, int key) {
    if (!root || root->val == key)
        return root;

    if (key < root->val)
        return search(root->left, key);

    return search(root->right, key);
}

// Iterative
TreeNode* searchIterative(TreeNode* root, int key) {
    while (root && root->val != key) {
        if (key < root->val)
            root = root->left;
        else
            root = root->right;
    }
    return root;
}
```

```python
def search(root, key):
    if not root or root.val == key:
        return root

    if key < root.val:
        return search(root.left, key)

    return search(root.right, key)
```

### 4.2 Insertion

```
Insert 5:
       (8)                 (8)
      /   \               /   \
    (3)   (10)    →     (3)   (10)
    / \                 / \
  (1)(6)              (1)(6)
                         /
                       (5)
```

```cpp
// C++
TreeNode* insert(TreeNode* root, int key) {
    if (!root) return new TreeNode(key);

    if (key < root->val)
        root->left = insert(root->left, key);
    else if (key > root->val)
        root->right = insert(root->right, key);

    return root;
}
```

```python
def insert(root, key):
    if not root:
        return TreeNode(key)

    if key < root.val:
        root.left = insert(root.left, key)
    elif key > root.val:
        root.right = insert(root.right, key)

    return root
```

### 4.3 Deletion

```
Three cases:

1. Leaf node: Just delete

2. One child: Replace with child

3. Two children: Replace with inorder successor
   - Minimum value in right subtree
   - Or maximum value in left subtree

       (8)                 (8)
      /   \               /   \
    (3)   (10)    →     (4)   (10)
    / \                 / \
  (1)(6)              (1)(6)
     /
   (4)

Delete 3: Replace with successor 4
```

```cpp
// C++
TreeNode* findMin(TreeNode* root) {
    while (root->left)
        root = root->left;
    return root;
}

TreeNode* deleteNode(TreeNode* root, int key) {
    if (!root) return nullptr;

    if (key < root->val) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->val) {
        root->right = deleteNode(root->right, key);
    } else {
        // Found!
        if (!root->left) {
            TreeNode* temp = root->right;
            delete root;
            return temp;
        }
        if (!root->right) {
            TreeNode* temp = root->left;
            delete root;
            return temp;
        }

        // Two children: Replace with successor
        TreeNode* successor = findMin(root->right);
        root->val = successor->val;
        root->right = deleteNode(root->right, successor->val);
    }

    return root;
}
```

```python
def delete_node(root, key):
    if not root:
        return None

    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left

        # Find successor
        successor = root.right
        while successor.left:
            successor = successor.left

        root.val = successor.val
        root.right = delete_node(root.right, successor.val)

    return root
```

---

## 5. Balanced Tree Concepts

### Skewed Tree Problem

```
1 → 2 → 3 → 4 → 5

All operations become O(n)!
```

### Types of Balanced Trees

```
1. AVL Tree
   - Height difference between left/right <= 1 for all nodes
   - Maintains balance through rotations on insert/delete

2. Red-Black Tree
   - Each node is red or black
   - Maintains balance through specific rules
   - Foundation for C++ map, set

3. B-Tree
   - Multi-way search tree
   - Used in databases
```

### Tree Height Calculation

```python
def height(root):
    if not root:
        return -1  # or 0
    return 1 + max(height(root.left), height(root.right))
```

### BST Validation

```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True

    if root.val <= min_val or root.val >= max_val:
        return False

    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))
```

---

## 6. Practice Problems

### Problem 1: Lowest Common Ancestor (LCA)

Find the lowest common ancestor of two nodes in a BST

<details>
<summary>Solution Code</summary>

```python
def lowest_common_ancestor(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root

    return None
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [Tree Traversal](https://www.acmicpc.net/problem/1991) | BOJ | Traversal |
| ⭐⭐ | [Validate BST](https://leetcode.com/problems/validate-binary-search-tree/) | LeetCode | BST |
| ⭐⭐ | [Binary Tree Inorder](https://leetcode.com/problems/binary-tree-inorder-traversal/) | LeetCode | Traversal |
| ⭐⭐ | [Binary Search Tree](https://www.acmicpc.net/problem/5639) | BOJ | BST |
| ⭐⭐⭐ | [LCA](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | LeetCode | LCA |

---

## Next Steps

- [10_Heaps_and_Priority_Queues.md](./10_Heaps_and_Priority_Queues.md) - Heaps, Priority Queues

---

## References

- [Tree Visualization](https://visualgo.net/en/bst)
- Introduction to Algorithms (CLRS) - Chapter 12, 13
