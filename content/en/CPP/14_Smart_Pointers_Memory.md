# Smart Pointers and Memory Management

## 1. Challenges of Memory Management

Manual memory management in C++ can cause several problems.

```cpp
#include <iostream>

// Memory leak example
void memoryLeak() {
    int* p = new int(42);
    // Forgot delete - memory leak!
}

// Double free example
void doubleFree() {
    int* p = new int(42);
    delete p;
    // delete p;  // Double free - undefined behavior!
}

// Dangling pointer example
int* danglingPointer() {
    int* p = new int(42);
    delete p;
    return p;  // Points to freed memory - dangerous!
}

// Memory leak on exception
void exceptionLeak() {
    int* p = new int(42);
    // throw std::runtime_error("Error!");  // delete won't execute
    delete p;
}
```

### Problem Summary

| Problem | Description |
|---------|-------------|
| Memory leak | Forgetting to call delete |
| Double free | Freeing the same memory twice |
| Dangling pointer | Accessing freed memory |
| Exception safety | Memory leak when exception occurs |

---

## 2. RAII (Resource Acquisition Is Initialization)

Resource Acquisition Is Initialization: Acquire resources at object creation, automatically release at destruction.

```cpp
#include <iostream>

// Class applying RAII principle
class IntPtr {
private:
    int* ptr;

public:
    // Acquire resource in constructor
    explicit IntPtr(int value) : ptr(new int(value)) {
        std::cout << "Memory allocated" << std::endl;
    }

    // Release resource in destructor
    ~IntPtr() {
        delete ptr;
        std::cout << "Memory freed" << std::endl;
    }

    int& operator*() { return *ptr; }
    int* get() { return ptr; }

    // Disable copy (simplified)
    IntPtr(const IntPtr&) = delete;
    IntPtr& operator=(const IntPtr&) = delete;
};

void useRAII() {
    IntPtr p(42);
    std::cout << "Value: " << *p << std::endl;
    // Memory automatically freed when function ends
}

int main() {
    std::cout << "=== RAII Start ===" << std::endl;
    useRAII();
    std::cout << "=== RAII End ===" << std::endl;
    return 0;
}
```

Output:
```
=== RAII Start ===
Memory allocated
Value: 42
Memory freed
=== RAII End ===
```

---

## 3. unique_ptr

A smart pointer with exclusive ownership. Only one `unique_ptr` can own an object.

### Basic Usage

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource created" << std::endl; }
    ~Resource() { std::cout << "Resource destroyed" << std::endl; }
    void use() { std::cout << "Resource used" << std::endl; }
};

int main() {
    // Create unique_ptr
    std::unique_ptr<Resource> p1(new Resource());
    p1->use();

    // Using make_unique (C++14, recommended)
    auto p2 = std::make_unique<Resource>();
    p2->use();

    // Basic type
    auto num = std::make_unique<int>(42);
    std::cout << "Value: " << *num << std::endl;

    // Array
    auto arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    std::cout << "Array: ";
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;  // All memory automatically freed
}
```

### Ownership Transfer (move)

```cpp
#include <iostream>
#include <memory>

void takeOwnership(std::unique_ptr<int> p) {
    std::cout << "Inside function: " << *p << std::endl;
}  // p is destroyed here

std::unique_ptr<int> createResource() {
    return std::make_unique<int>(100);
}

int main() {
    auto p1 = std::make_unique<int>(42);

    // Cannot copy
    // auto p2 = p1;  // Compile error!

    // Move is allowed
    auto p2 = std::move(p1);
    std::cout << "p2: " << *p2 << std::endl;

    // p1 is now nullptr
    if (p1 == nullptr) {
        std::cout << "p1 is empty" << std::endl;
    }

    // Pass to function (ownership transfer)
    auto p3 = std::make_unique<int>(200);
    takeOwnership(std::move(p3));
    // p3 is now nullptr

    // Return from function (ownership transfer)
    auto p4 = createResource();
    std::cout << "p4: " << *p4 << std::endl;

    return 0;
}
```

### unique_ptr Methods

```cpp
#include <iostream>
#include <memory>

int main() {
    auto p = std::make_unique<int>(42);

    // get(): Get raw pointer (ownership retained)
    int* raw = p.get();
    std::cout << "raw: " << *raw << std::endl;

    // release(): Give up ownership and return raw pointer
    int* released = p.release();
    if (p == nullptr) {
        std::cout << "p is empty" << std::endl;
    }
    delete released;  // Manual deletion needed

    // reset(): Release existing object and set new one
    auto p2 = std::make_unique<int>(100);
    std::cout << "Before reset: " << *p2 << std::endl;
    p2.reset(new int(200));
    std::cout << "After reset: " << *p2 << std::endl;
    p2.reset();  // Set to nullptr
    if (!p2) {
        std::cout << "p2 is empty" << std::endl;
    }

    // swap(): Exchange two pointers
    auto a = std::make_unique<int>(1);
    auto b = std::make_unique<int>(2);
    a.swap(b);
    std::cout << "After swap: a=" << *a << ", b=" << *b << std::endl;

    return 0;
}
```

### Custom Deleter

```cpp
#include <iostream>
#include <memory>
#include <cstdio>

// Function deleter
void customDeleter(int* p) {
    std::cout << "Custom deleter called" << std::endl;
    delete p;
}

// Deleter for FILE*
auto fileDeleter = [](FILE* f) {
    if (f) {
        std::cout << "Closing file" << std::endl;
        fclose(f);
    }
};

int main() {
    // Function pointer deleter
    std::unique_ptr<int, void(*)(int*)> p1(
        new int(42), customDeleter
    );

    // Lambda deleter
    auto deleter = [](int* p) {
        std::cout << "Lambda deleter" << std::endl;
        delete p;
    };
    std::unique_ptr<int, decltype(deleter)> p2(
        new int(100), deleter
    );

    // FILE management
    std::unique_ptr<FILE, decltype(fileDeleter)> file(
        fopen("test.txt", "w"), fileDeleter
    );
    if (file) {
        fprintf(file.get(), "Hello, World!\n");
    }

    return 0;
}
```

---

## 4. shared_ptr

A smart pointer with shared ownership. Multiple `shared_ptr`s can share the same object.

### Basic Usage

```cpp
#include <iostream>
#include <memory>

class Resource {
public:
    Resource() { std::cout << "Resource created" << std::endl; }
    ~Resource() { std::cout << "Resource destroyed" << std::endl; }
};

int main() {
    // Create shared_ptr
    std::shared_ptr<Resource> p1 = std::make_shared<Resource>();
    std::cout << "Reference count: " << p1.use_count() << std::endl;  // 1

    {
        // Share
        std::shared_ptr<Resource> p2 = p1;
        std::cout << "Reference count: " << p1.use_count() << std::endl;  // 2

        std::shared_ptr<Resource> p3 = p1;
        std::cout << "Reference count: " << p1.use_count() << std::endl;  // 3
    }
    // p2, p3 destroyed
    std::cout << "Reference count: " << p1.use_count() << std::endl;  // 1

    return 0;  // Resource destroyed when reference count becomes 0
}
```

### Advantages of make_shared

```cpp
#include <iostream>
#include <memory>

class Widget {
public:
    int data[100];
};

int main() {
    // Method 1: Using new (2 memory allocations)
    std::shared_ptr<Widget> p1(new Widget());

    // Method 2: Using make_shared (1 memory allocation, recommended)
    auto p2 = std::make_shared<Widget>();

    /*
    Advantages of make_shared:
    1. Single memory allocation (object + control block)
    2. Exception safety
    3. Cleaner code
    */

    std::cout << "p1 use_count: " << p1.use_count() << std::endl;
    std::cout << "p2 use_count: " << p2.use_count() << std::endl;

    return 0;
}
```

### shared_ptr and Containers

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Person {
public:
    std::string name;
    Person(const std::string& n) : name(n) {
        std::cout << name << " created" << std::endl;
    }
    ~Person() {
        std::cout << name << " destroyed" << std::endl;
    }
};

int main() {
    std::vector<std::shared_ptr<Person>> people;

    auto alice = std::make_shared<Person>("Alice");
    auto bob = std::make_shared<Person>("Bob");

    people.push_back(alice);
    people.push_back(bob);
    people.push_back(alice);  // Alice shared

    std::cout << "Alice reference count: " << alice.use_count() << std::endl;  // 3

    std::cout << "\n=== List ===" << std::endl;
    for (const auto& p : people) {
        std::cout << p->name << std::endl;
    }

    people.clear();
    std::cout << "\n=== After clear ===" << std::endl;
    std::cout << "Alice reference count: " << alice.use_count() << std::endl;  // 1

    return 0;
}
```

---

## 5. weak_ptr

Solves the circular reference problem of `shared_ptr`. Does not increment the reference count.

### Circular Reference Problem

```cpp
#include <iostream>
#include <memory>

class B;  // Forward declaration

class A {
public:
    std::shared_ptr<B> b_ptr;

    ~A() { std::cout << "A destroyed" << std::endl; }
};

class B {
public:
    std::shared_ptr<A> a_ptr;  // Circular reference!

    ~B() { std::cout << "B destroyed" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // Circular reference occurs

        std::cout << "a reference count: " << a.use_count() << std::endl;  // 2
        std::cout << "b reference count: " << b.use_count() << std::endl;  // 2
    }
    // Memory leak! Neither A nor B is destroyed
    std::cout << "Block ended" << std::endl;

    return 0;
}
```

### Solution with weak_ptr

```cpp
#include <iostream>
#include <memory>

class B;

class A {
public:
    std::shared_ptr<B> b_ptr;

    ~A() { std::cout << "A destroyed" << std::endl; }
};

class B {
public:
    std::weak_ptr<A> a_ptr;  // Using weak_ptr!

    ~B() { std::cout << "B destroyed" << std::endl; }
};

int main() {
    {
        auto a = std::make_shared<A>();
        auto b = std::make_shared<B>();

        a->b_ptr = b;
        b->a_ptr = a;  // weak_ptr doesn't increment reference count

        std::cout << "a reference count: " << a.use_count() << std::endl;  // 1
        std::cout << "b reference count: " << b.use_count() << std::endl;  // 2
    }
    // Properly destroyed!
    std::cout << "Block ended" << std::endl;

    return 0;
}
```

### weak_ptr Usage

```cpp
#include <iostream>
#include <memory>

int main() {
    std::weak_ptr<int> weak;

    {
        auto shared = std::make_shared<int>(42);
        weak = shared;

        std::cout << "Inside block:" << std::endl;
        std::cout << "  expired: " << weak.expired() << std::endl;  // false
        std::cout << "  use_count: " << weak.use_count() << std::endl;  // 1

        // Accessing weak_ptr: Get shared_ptr with lock()
        if (auto sp = weak.lock()) {
            std::cout << "  Value: " << *sp << std::endl;
        }
    }
    // shared is destroyed

    std::cout << "Outside block:" << std::endl;
    std::cout << "  expired: " << weak.expired() << std::endl;  // true
    std::cout << "  use_count: " << weak.use_count() << std::endl;  // 0

    if (auto sp = weak.lock()) {
        std::cout << "  Value: " << *sp << std::endl;
    } else {
        std::cout << "  Object is destroyed" << std::endl;
    }

    return 0;
}
```

### Cache Implementation Example

```cpp
#include <iostream>
#include <memory>
#include <map>
#include <string>

class Image {
public:
    std::string filename;

    Image(const std::string& fn) : filename(fn) {
        std::cout << "Loading image: " << filename << std::endl;
    }
    ~Image() {
        std::cout << "Releasing image: " << filename << std::endl;
    }
};

class ImageCache {
private:
    std::map<std::string, std::weak_ptr<Image>> cache;

public:
    std::shared_ptr<Image> getImage(const std::string& filename) {
        auto it = cache.find(filename);

        if (it != cache.end()) {
            // If in cache, try to get shared_ptr from weak_ptr
            if (auto sp = it->second.lock()) {
                std::cout << "Cache hit: " << filename << std::endl;
                return sp;
            }
        }

        // Cache miss: Load new
        std::cout << "Cache miss: " << filename << std::endl;
        auto image = std::make_shared<Image>(filename);
        cache[filename] = image;
        return image;
    }
};

int main() {
    ImageCache cache;

    {
        auto img1 = cache.getImage("photo.jpg");
        auto img2 = cache.getImage("photo.jpg");  // Cache hit
        auto img3 = cache.getImage("icon.png");

        std::cout << "img1 use_count: " << img1.use_count() << std::endl;
    }
    // All images released

    std::cout << "\n=== Request again ===" << std::endl;
    auto img = cache.getImage("photo.jpg");  // Load again

    return 0;
}
```

---

## 6. enable_shared_from_this

Safely get a `shared_ptr` of yourself from within a class.

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Task : public std::enable_shared_from_this<Task> {
public:
    std::string name;

    Task(const std::string& n) : name(n) {
        std::cout << name << " created" << std::endl;
    }

    ~Task() {
        std::cout << name << " destroyed" << std::endl;
    }

    // Safely return shared_ptr to self
    std::shared_ptr<Task> getPtr() {
        return shared_from_this();
    }

    void addToQueue(std::vector<std::shared_ptr<Task>>& queue) {
        queue.push_back(shared_from_this());
    }
};

int main() {
    std::vector<std::shared_ptr<Task>> taskQueue;

    {
        auto task = std::make_shared<Task>("Task1");
        std::cout << "Reference count: " << task.use_count() << std::endl;  // 1

        task->addToQueue(taskQueue);
        std::cout << "Reference count: " << task.use_count() << std::endl;  // 2
    }
    // task variable destroyed, but remains in taskQueue

    std::cout << "\n=== Queue contents ===" << std::endl;
    for (const auto& t : taskQueue) {
        std::cout << t->name << std::endl;
    }

    return 0;
}
```

Caution:
```cpp
// Wrong usage - must be managed by shared_ptr
// Task t("Direct");
// t.getPtr();  // Runtime error!
```

---

## 7. Smart Pointer Selection Guide

```
┌─────────────────────────────────────────────────────┐
│              Smart Pointer Selection                │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Exclusive        Shared         Weak
    Ownership?      Needed?       Reference?
          │             │             │
          ▼             ▼             ▼
    unique_ptr    shared_ptr     weak_ptr
```

| Situation | Choice |
|-----------|--------|
| Single owner | `unique_ptr` |
| Multiple owners | `shared_ptr` |
| Prevent circular reference | `weak_ptr` |
| Cache, Observer | `weak_ptr` |
| Factory function return | `unique_ptr` |
| Container storage | `shared_ptr` or `unique_ptr` |

---

## 8. Smart Pointers and Functions

### Function Parameters

```cpp
#include <iostream>
#include <memory>

class Widget {
public:
    int value;
    Widget(int v) : value(v) {}
};

// Transfer ownership (unique_ptr)
void takeOwnership(std::unique_ptr<Widget> w) {
    std::cout << "Ownership received: " << w->value << std::endl;
}

// Share ownership (shared_ptr copy)
void shareOwnership(std::shared_ptr<Widget> w) {
    std::cout << "Shared: " << w->value
              << " (count: " << w.use_count() << ")" << std::endl;
}

// Use without ownership (reference)
void useOnly(Widget& w) {
    std::cout << "Use only: " << w.value << std::endl;
}

// Use without ownership (raw pointer)
void useOnlyPtr(Widget* w) {
    if (w) {
        std::cout << "Pointer use: " << w->value << std::endl;
    }
}

int main() {
    // unique_ptr
    auto up = std::make_unique<Widget>(1);
    useOnly(*up);
    useOnlyPtr(up.get());
    takeOwnership(std::move(up));  // Transfer ownership

    // shared_ptr
    auto sp = std::make_shared<Widget>(2);
    useOnly(*sp);
    useOnlyPtr(sp.get());
    shareOwnership(sp);  // Share
    std::cout << "Original count: " << sp.use_count() << std::endl;

    return 0;
}
```

### Function Return

```cpp
#include <iostream>
#include <memory>

class Product {
public:
    std::string name;
    Product(const std::string& n) : name(n) {}
};

// Factory function: Return unique_ptr
std::unique_ptr<Product> createProduct(const std::string& name) {
    return std::make_unique<Product>(name);
}

// Cached object: Return shared_ptr
std::shared_ptr<Product> getCachedProduct() {
    static auto cached = std::make_shared<Product>("Cached");
    return cached;
}

int main() {
    auto p1 = createProduct("Widget");
    std::cout << p1->name << std::endl;

    auto p2 = getCachedProduct();
    auto p3 = getCachedProduct();
    std::cout << "Cache count: " << p2.use_count() << std::endl;  // 3

    return 0;
}
```

---

## 9. Common Mistakes and Solutions

### Mistake 1: Creating Multiple Smart Pointers from Same Raw Pointer

```cpp
#include <iostream>
#include <memory>

int main() {
    int* raw = new int(42);

    // Wrong code - never do this!
    // std::shared_ptr<int> p1(raw);
    // std::shared_ptr<int> p2(raw);  // Double free!

    // Correct code
    auto p1 = std::make_shared<int>(42);
    auto p2 = p1;  // Share

    return 0;
}
```

### Mistake 2: Converting this to shared_ptr

```cpp
#include <iostream>
#include <memory>

class Bad {
public:
    // Wrong method
    std::shared_ptr<Bad> getShared() {
        // return std::shared_ptr<Bad>(this);  // Dangerous!
        return nullptr;
    }
};

class Good : public std::enable_shared_from_this<Good> {
public:
    // Correct method
    std::shared_ptr<Good> getShared() {
        return shared_from_this();
    }
};
```

### Mistake 3: Circular Reference

```cpp
// See weak_ptr section above
// Using only shared_ptr causes memory leak due to circular reference
// Change one connection to weak reference using weak_ptr
```

### Mistake 4: Attempting to Copy unique_ptr

```cpp
#include <memory>

void processWidget(std::unique_ptr<int> p) {}

int main() {
    auto p = std::make_unique<int>(42);

    // Wrong code
    // processWidget(p);  // Compile error

    // Correct code (ownership transfer)
    processWidget(std::move(p));

    return 0;
}
```

---

## 10. Performance Considerations

### unique_ptr vs shared_ptr

```cpp
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

int main() {
    const int N = 1000000;

    // unique_ptr (almost no overhead)
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto p = std::make_unique<int>(i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    // shared_ptr (reference counting overhead)
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto p = std::make_shared<int>(i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

    std::cout << "unique_ptr: " << dur1.count() << " us" << std::endl;
    std::cout << "shared_ptr: " << dur2.count() << " us" << std::endl;

    return 0;
}
```

### Memory Structure

```
unique_ptr:
┌─────────────────┐
│  ptr → Object   │  (only one pointer)
└─────────────────┘

shared_ptr:
┌─────────────────┐     ┌─────────────────┐
│  ptr ─────────────┬──▶│     Object      │
│  control ───┐    │    └─────────────────┘
└─────────────│───┘
              ▼
        ┌─────────────────┐
        │  Reference count│
        │  Weak count     │
        │  Deleter        │
        └─────────────────┘
```

---

## 11. Summary

| Smart Pointer | Ownership | Copy | Ref Count | Use Case |
|---------------|-----------|------|-----------|----------|
| `unique_ptr` | Exclusive | X | X | Single owner |
| `shared_ptr` | Shared | O | O | Shared ownership |
| `weak_ptr` | None | O | X | Prevent circular reference |

### Core Principles

1. **Avoid direct new/delete** - Use `make_unique`, `make_shared`
2. **Default to unique_ptr** - Only use shared_ptr when needed
3. **Beware of circular references** - Solve with weak_ptr
4. **Follow RAII principle** - Automate resource management

---

## 12. Exercises

### Exercise 1: Resource Manager

Implement a class that manages various resources (file, network connection, etc.) using `unique_ptr`.

### Exercise 2: Graph Data Structure

Implement a graph where nodes are connected to each other using `shared_ptr` and `weak_ptr`.

### Exercise 3: Object Pool

Implement a reusable object pool using smart pointers.

---

## Next Step

Let's learn about C++11/14/17/20 major features in [15_Modern_CPP.md](./15_Modern_CPP.md)!
