# STL Algorithms and Iterators

## 1. Iterator

Iterators are pointer-like objects that point to container elements.

### Iterator Types

| Type | Description | Example Container |
|------|-------------|------------------|
| Input Iterator | Read-only, one direction | istream_iterator |
| Output Iterator | Write-only, one direction | ostream_iterator |
| Forward Iterator | Read/write, one direction | forward_list |
| Bidirectional Iterator | Read/write, both directions | list, set, map |
| Random Access Iterator | All operations, random access | vector, deque, array |

### Basic Iterator Usage

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // begin(), end()
    std::vector<int>::iterator it = v.begin();
    std::cout << *it << std::endl;  // 1

    ++it;
    std::cout << *it << std::endl;  // 2

    // Iteration
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### const Iterator

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // const_iterator: read-only
    for (std::vector<int>::const_iterator it = v.cbegin();
         it != v.cend(); ++it) {
        std::cout << *it << " ";
        // *it = 10;  // Error! Cannot modify
    }

    return 0;
}
```

### Reverse Iterator

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // rbegin(), rend()
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

---

## 2. Lambda Expressions

Defines anonymous functions concisely.

### Basic Syntax

```cpp
[capture](parameters) -> return_type { body }
```

### Examples

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    // Basic lambda
    auto add = [](int a, int b) {
        return a + b;
    };
    std::cout << add(3, 5) << std::endl;  // 8

    // Explicit return type
    auto divide = [](double a, double b) -> double {
        return a / b;
    };

    // With algorithms
    std::vector<int> v = {3, 1, 4, 1, 5, 9};
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;  // Descending order
    });

    return 0;
}
```

### Capture

```cpp
#include <iostream>

int main() {
    int x = 10;
    int y = 20;

    // Capture by value (copy)
    auto f1 = [x]() { return x; };

    // Capture by reference
    auto f2 = [&x]() { x++; };

    // Capture all by value
    auto f3 = [=]() { return x + y; };

    // Capture all by reference
    auto f4 = [&]() { x++; y++; };

    // Mixed
    auto f5 = [=, &x]() {  // y by value, x by reference
        x++;
        return y;
    };

    f2();
    std::cout << x << std::endl;  // 11

    return 0;
}
```

### mutable Lambda

```cpp
#include <iostream>

int main() {
    int x = 10;

    // Value capture is const by default
    auto f = [x]() mutable {  // mutable allows modification
        x++;
        return x;
    };

    std::cout << f() << std::endl;  // 11
    std::cout << x << std::endl;    // 10 (original unchanged)

    return 0;
}
```

---

## 3. Basic Algorithms

Include the `<algorithm>` header.

### for_each

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    std::for_each(v.begin(), v.end(), [](int n) {
        std::cout << n * 2 << " ";
    });
    std::cout << std::endl;  // 2 4 6 8 10

    return 0;
}
```

### transform

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> result(v.size());

    // Transform each element
    std::transform(v.begin(), v.end(), result.begin(),
                   [](int n) { return n * n; });

    for (int n : result) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 4 9 16 25

    return 0;
}
```

---

## 4. Search Algorithms

### find

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    auto it = std::find(v.begin(), v.end(), 3);
    if (it != v.end()) {
        std::cout << "Found: " << *it << std::endl;
        std::cout << "Index: " << std::distance(v.begin(), it) << std::endl;
    }

    return 0;
}
```

### find_if

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // First element satisfying condition
    auto it = std::find_if(v.begin(), v.end(),
                           [](int n) { return n > 3; });

    if (it != v.end()) {
        std::cout << "First > 3: " << *it << std::endl;  // 4
    }

    return 0;
}
```

### count / count_if

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 2, 3, 2, 4, 5};

    // Count specific value
    int c1 = std::count(v.begin(), v.end(), 2);
    std::cout << "Count of 2: " << c1 << std::endl;  // 3

    // Count satisfying condition
    int c2 = std::count_if(v.begin(), v.end(),
                           [](int n) { return n % 2 == 0; });
    std::cout << "Even count: " << c2 << std::endl;  // 4

    return 0;
}
```

### binary_search

Use only on sorted ranges.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};  // Sorted

    bool found = std::binary_search(v.begin(), v.end(), 3);
    std::cout << "3 found: " << found << std::endl;  // 1

    return 0;
}
```

---

## 5. Sorting Algorithms

### sort

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Ascending (default)
    std::sort(v.begin(), v.end());

    // Descending
    std::sort(v.begin(), v.end(), std::greater<int>());

    // Custom comparison
    std::sort(v.begin(), v.end(), [](int a, int b) {
        return a > b;  // Descending
    });

    return 0;
}
```

### partial_sort

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Sort only top 3
    std::partial_sort(v.begin(), v.begin() + 3, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    // 1 1 2 ... (only first 3 sorted)

    return 0;
}
```

### nth_element

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Place 3rd element in its sorted position
    std::nth_element(v.begin(), v.begin() + 3, v.end());

    std::cout << "3rd element: " << v[3] << std::endl;

    return 0;
}
```

---

## 6. Modifying Algorithms

### copy

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dest(5);

    std::copy(src.begin(), src.end(), dest.begin());

    return 0;
}
```

### fill

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v(5);

    std::fill(v.begin(), v.end(), 42);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 42 42 42 42 42

    return 0;
}
```

### replace

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 2, 4, 2, 5};

    // Replace 2 with 100
    std::replace(v.begin(), v.end(), 2, 100);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 100 3 100 4 100 5

    return 0;
}
```

### remove / erase

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 2, 4, 2, 5};

    // remove doesn't actually delete
    auto newEnd = std::remove(v.begin(), v.end(), 2);

    // Use with erase (erase-remove idiom)
    v.erase(newEnd, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 3 4 5

    return 0;
}
```

### reverse

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    std::reverse(v.begin(), v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 5 4 3 2 1

    return 0;
}
```

### unique

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 1, 2, 2, 2, 3, 3, 4};

    // Remove consecutive duplicates (requires sorted)
    auto newEnd = std::unique(v.begin(), v.end());
    v.erase(newEnd, v.end());

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 2 3 4

    return 0;
}
```

---

## 7. Numeric Algorithms

Include the `<numeric>` header.

### accumulate

```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // Sum
    int sum = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "Sum: " << sum << std::endl;  // 15

    // Product
    int product = std::accumulate(v.begin(), v.end(), 1,
                                  std::multiplies<int>());
    std::cout << "Product: " << product << std::endl;  // 120

    // Custom
    int sumSquares = std::accumulate(v.begin(), v.end(), 0,
        [](int acc, int n) { return acc + n * n; });
    std::cout << "Sum of squares: " << sumSquares << std::endl;  // 55

    return 0;
}
```

### iota

```cpp
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    std::vector<int> v(10);

    // Fill with consecutive values
    std::iota(v.begin(), v.end(), 1);

    for (int n : v) {
        std::cout << n << " ";
    }
    std::cout << std::endl;  // 1 2 3 4 5 6 7 8 9 10

    return 0;
}
```

---

## 8. Set Algorithms

Works only on sorted ranges.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {3, 4, 5, 6, 7};
    std::vector<int> result;

    // Union
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(result));
    // result: 1 2 3 4 5 6 7

    result.clear();

    // Intersection
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(result));
    // result: 3 4 5

    result.clear();

    // Difference
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
    // result: 1 2

    return 0;
}
```

---

## 9. min/max Algorithms

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // Min/max element
    auto minIt = std::min_element(v.begin(), v.end());
    auto maxIt = std::max_element(v.begin(), v.end());

    std::cout << "Min: " << *minIt << std::endl;
    std::cout << "Max: " << *maxIt << std::endl;

    // Both
    auto [minEl, maxEl] = std::minmax_element(v.begin(), v.end());
    std::cout << *minEl << " ~ " << *maxEl << std::endl;

    // Value comparison
    std::cout << std::min(3, 5) << std::endl;  // 3
    std::cout << std::max(3, 5) << std::endl;  // 5

    return 0;
}
```

---

## 10. all_of / any_of / none_of

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {2, 4, 6, 8, 10};

    // All satisfy?
    bool all = std::all_of(v.begin(), v.end(),
                           [](int n) { return n % 2 == 0; });
    std::cout << "All even: " << all << std::endl;  // 1

    // Any satisfy?
    bool any = std::any_of(v.begin(), v.end(),
                           [](int n) { return n > 5; });
    std::cout << "Any > 5: " << any << std::endl;  // 1

    // None satisfy?
    bool none = std::none_of(v.begin(), v.end(),
                             [](int n) { return n < 0; });
    std::cout << "No negatives: " << none << std::endl;  // 1

    return 0;
}
```

---

## 11. Summary

| Algorithm | Purpose |
|-----------|---------|
| `find`, `find_if` | Search |
| `count`, `count_if` | Count |
| `sort`, `partial_sort` | Sort |
| `binary_search` | Binary search |
| `transform` | Transform |
| `for_each` | Apply function to each element |
| `copy`, `fill`, `replace` | Modify |
| `remove`, `unique` | Remove |
| `reverse` | Reverse |
| `accumulate` | Accumulate |
| `min_element`, `max_element` | Min/max |

---

## Next Steps

Let's learn about templates in [12_Templates.md](./12_Templates.md)!
