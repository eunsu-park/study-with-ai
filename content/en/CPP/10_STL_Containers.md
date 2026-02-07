# STL Containers

## 1. What is STL?

STL (Standard Template Library) is the core of the C++ standard library, providing data structures and algorithms.

### STL Components

| Component | Description |
|-----------|-------------|
| Containers | Data structures for storing data |
| Iterators | Traversing container elements |
| Algorithms | General-purpose functions like sorting, searching |
| Function Objects | Objects that behave like functions |

---

## 2. vector

A dynamic-sized array. Most commonly used.

### Basic Usage

```cpp
#include <iostream>
#include <vector>

int main() {
    // Creation
    std::vector<int> v1;                  // Empty vector
    std::vector<int> v2(5);               // Size 5, initialized to 0
    std::vector<int> v3(5, 10);           // Size 5, initialized to 10
    std::vector<int> v4 = {1, 2, 3, 4, 5}; // Initializer list

    // Adding elements
    v1.push_back(10);
    v1.push_back(20);
    v1.push_back(30);

    // Output
    for (int num : v1) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

### Element Access

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {10, 20, 30, 40, 50};

    // Index access
    std::cout << v[0] << std::endl;      // 10
    std::cout << v.at(2) << std::endl;   // 30 (range checked)

    // First/last
    std::cout << v.front() << std::endl;  // 10
    std::cout << v.back() << std::endl;   // 50

    // Size
    std::cout << "Size: " << v.size() << std::endl;
    std::cout << "Empty: " << v.empty() << std::endl;

    return 0;
}
```

### Insertion and Deletion

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // Add/remove at end
    v.push_back(6);   // {1, 2, 3, 4, 5, 6}
    v.pop_back();     // {1, 2, 3, 4, 5}

    // Insert in middle
    v.insert(v.begin() + 2, 100);  // {1, 2, 100, 3, 4, 5}

    // Delete in middle
    v.erase(v.begin() + 2);  // {1, 2, 3, 4, 5}

    // Range deletion
    v.erase(v.begin(), v.begin() + 2);  // {3, 4, 5}

    // Clear all
    v.clear();

    return 0;
}
```

### Iterators

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // Iterate with iterator
    for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // Using auto (recommended)
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // Reverse iterator
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

---

## 3. array

A fixed-size array.

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // Access
    std::cout << arr[0] << std::endl;
    std::cout << arr.at(2) << std::endl;
    std::cout << arr.front() << std::endl;
    std::cout << arr.back() << std::endl;

    // Size
    std::cout << "Size: " << arr.size() << std::endl;

    // Iterate
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Fill
    arr.fill(0);

    return 0;
}
```

---

## 4. deque

A container with fast insertion/deletion at both ends.

```cpp
#include <iostream>
#include <deque>

int main() {
    std::deque<int> dq;

    // Add to front/back
    dq.push_back(1);
    dq.push_back(2);
    dq.push_front(0);
    dq.push_front(-1);

    // {-1, 0, 1, 2}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Remove from front/back
    dq.pop_front();
    dq.pop_back();

    // {0, 1}
    for (int num : dq) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

## 5. list

A doubly-linked list.

```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};

    // Add to front/back
    lst.push_front(0);
    lst.push_back(9);

    // Sort (member method)
    lst.sort();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 1 3 4 5 9

    // Remove duplicates (consecutive only)
    lst.unique();

    for (int num : lst) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 0 1 3 4 5 9

    // Insert
    auto it = lst.begin();
    std::advance(it, 2);  // Move 2 positions
    lst.insert(it, 100);  // Insert at that position

    return 0;
}
```

---

## 6. set

A sorted collection of unique elements.

```cpp
#include <iostream>
#include <set>

int main() {
    std::set<int> s;

    // Insert
    s.insert(30);
    s.insert(10);
    s.insert(20);
    s.insert(10);  // Duplicate, ignored

    // Auto-sorted
    for (int num : s) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 20 30

    // Size
    std::cout << "Size: " << s.size() << std::endl;  // 3

    // Search
    if (s.find(20) != s.end()) {
        std::cout << "20 found" << std::endl;
    }

    // count (0 or 1)
    std::cout << "Count of 30: " << s.count(30) << std::endl;

    // Delete
    s.erase(20);

    return 0;
}
```

### multiset

A set that allows duplicates.

```cpp
#include <iostream>
#include <set>

int main() {
    std::multiset<int> ms;

    ms.insert(10);
    ms.insert(10);
    ms.insert(20);
    ms.insert(10);

    for (int num : ms) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // 10 10 10 20

    std::cout << "Count of 10: " << ms.count(10) << std::endl;  // 3

    return 0;
}
```

---

## 7. map

A sorted container of key-value pairs.

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> ages;

    // Insert
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    ages.insert({"Charlie", 35});
    ages.insert(std::make_pair("David", 40));

    // Access
    std::cout << "Alice: " << ages["Alice"] << std::endl;

    // Iterate (sorted by key)
    for (const auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Structured binding (C++17)
    for (const auto& [name, age] : ages) {
        std::cout << name << ": " << age << std::endl;
    }

    // Search
    if (ages.find("Alice") != ages.end()) {
        std::cout << "Alice found" << std::endl;
    }

    // Delete
    ages.erase("Bob");

    return 0;
}
```

### Note: operator[]

```cpp
std::map<std::string, int> m;

// Accessing non-existent key → inserts with default value (0)!
std::cout << m["unknown"] << std::endl;  // 0 (and gets inserted)
std::cout << m.size() << std::endl;      // 1

// Safe access
if (m.count("key") > 0) {
    std::cout << m["key"] << std::endl;
}

// Or use find
auto it = m.find("key");
if (it != m.end()) {
    std::cout << it->second << std::endl;
}
```

---

## 8. unordered_set / unordered_map

Hash table-based containers with average O(1) access.

### unordered_set

```cpp
#include <iostream>
#include <unordered_set>

int main() {
    std::unordered_set<int> us;

    us.insert(30);
    us.insert(10);
    us.insert(20);

    // Order not guaranteed
    for (int num : us) {
        std::cout << num << " ";
    }
    std::cout << std::endl;  // Order undefined

    // Search (O(1) average)
    if (us.count(20)) {
        std::cout << "20 found" << std::endl;
    }

    return 0;
}
```

### unordered_map

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::unordered_map<std::string, int> umap;

    umap["apple"] = 100;
    umap["banana"] = 200;
    umap["cherry"] = 300;

    // Access (O(1) average)
    std::cout << "apple: " << umap["apple"] << std::endl;

    // Iterate (order undefined)
    for (const auto& [key, value] : umap) {
        std::cout << key << ": " << value << std::endl;
    }

    return 0;
}
```

### set vs unordered_set

| Feature | set | unordered_set |
|---------|-----|---------------|
| Internal structure | Red-black tree | Hash table |
| Sorted | Yes | No |
| Insert/search | O(log n) | O(1) average |
| Iteration order | Sorted | Undefined |

---

## 9. stack and queue

Container adapters.

### stack (LIFO)

```cpp
#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;

    // push
    s.push(10);
    s.push(20);
    s.push(30);

    // pop (LIFO)
    while (!s.empty()) {
        std::cout << s.top() << " ";  // Top element
        s.pop();
    }
    std::cout << std::endl;  // 30 20 10

    return 0;
}
```

### queue (FIFO)

```cpp
#include <iostream>
#include <queue>

int main() {
    std::queue<int> q;

    // push
    q.push(10);
    q.push(20);
    q.push(30);

    // pop (FIFO)
    while (!q.empty()) {
        std::cout << q.front() << " ";  // Front element
        q.pop();
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

### priority_queue

```cpp
#include <iostream>
#include <queue>

int main() {
    // Default: max heap (larger values first)
    std::priority_queue<int> pq;

    pq.push(30);
    pq.push(10);
    pq.push(20);

    while (!pq.empty()) {
        std::cout << pq.top() << " ";
        pq.pop();
    }
    std::cout << std::endl;  // 30 20 10

    // Min heap
    std::priority_queue<int, std::vector<int>, std::greater<int>> minPq;

    minPq.push(30);
    minPq.push(10);
    minPq.push(20);

    while (!minPq.empty()) {
        std::cout << minPq.top() << " ";
        minPq.pop();
    }
    std::cout << std::endl;  // 10 20 30

    return 0;
}
```

---

## 10. pair and tuple

### pair

```cpp
#include <iostream>
#include <utility>

int main() {
    // Creation
    std::pair<std::string, int> p1("Alice", 25);
    auto p2 = std::make_pair("Bob", 30);

    // Access
    std::cout << p1.first << ": " << p1.second << std::endl;

    // Comparison
    if (p1 < p2) {  // Compares first, then second
        std::cout << p1.first << " < " << p2.first << std::endl;
    }

    return 0;
}
```

### tuple

```cpp
#include <iostream>
#include <tuple>
#include <string>

int main() {
    // Creation
    std::tuple<std::string, int, double> t("Alice", 25, 165.5);

    // Access
    std::cout << std::get<0>(t) << std::endl;  // Alice
    std::cout << std::get<1>(t) << std::endl;  // 25
    std::cout << std::get<2>(t) << std::endl;  // 165.5

    // Structured binding (C++17)
    auto [name, age, height] = t;
    std::cout << name << ", " << age << ", " << height << std::endl;

    return 0;
}
```

---

## 11. Container Selection Guide

| Requirement | Recommended Container |
|-------------|----------------------|
| Sequential access + end insertion/deletion | `vector` |
| Both ends insertion/deletion | `deque` |
| Frequent middle insertion/deletion | `list` |
| Unique elements + sorted | `set` |
| Unique elements + fast search | `unordered_set` |
| Key-value + sorted | `map` |
| Key-value + fast search | `unordered_map` |
| LIFO | `stack` |
| FIFO | `queue` |
| Priority | `priority_queue` |

---

## 12. Summary

| Container | Characteristics |
|-----------|----------------|
| `vector` | Dynamic array, O(1) at end |
| `array` | Fixed array |
| `deque` | O(1) at both ends |
| `list` | Doubly-linked list |
| `set` | Sorted + unique |
| `map` | Key-value + sorted |
| `unordered_set` | Hash + unique |
| `unordered_map` | Hash + key-value |
| `stack` | LIFO |
| `queue` | FIFO |
| `priority_queue` | Heap |

---

## Next Steps

Let's learn about STL algorithms in [11_STL_Algorithms_Iterators.md](./11_STL_Algorithms_Iterators.md)!
