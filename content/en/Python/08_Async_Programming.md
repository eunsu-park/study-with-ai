# Async Programming

## 1. Synchronous vs Asynchronous

### Synchronous

Tasks execute sequentially, one must finish before the next begins.

```python
import time

def task(name, duration):
    print(f"{name} starting")
    time.sleep(duration)  # Blocking
    print(f"{name} done")

# Total 6 seconds
task("Task1", 2)
task("Task2", 2)
task("Task3", 2)
```

### Asynchronous

Other tasks can run during I/O waits.

```python
import asyncio

async def task(name, duration):
    print(f"{name} starting")
    await asyncio.sleep(duration)  # Non-blocking
    print(f"{name} done")

async def main():
    # Concurrent execution - total 2 seconds
    await asyncio.gather(
        task("Task1", 2),
        task("Task2", 2),
        task("Task3", 2)
    )

asyncio.run(main())
```

### Comparison Diagram

```
Synchronous execution:
Task1: ████████
Task2:         ████████
Task3:                  ████████
Time:  0       2        4        6s

Asynchronous execution (I/O bound):
Task1: ████████
Task2: ████████
Task3: ████████
Time:  0       2s
```

---

## 2. async/await Basics

### Defining Coroutines

```python
async def my_coroutine():
    return "Hello, Async!"

# Calling coroutine → returns coroutine object
coro = my_coroutine()
print(coro)  # <coroutine object my_coroutine at ...>

# To execute, need await or asyncio.run()
result = asyncio.run(my_coroutine())
print(result)  # Hello, Async!
```

### The await Keyword

`await` waits for coroutines, Tasks, or Futures.

```python
async def fetch_data():
    print("Fetching data...")
    await asyncio.sleep(1)  # I/O simulation
    return {"data": "value"}

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

### Important: await Only Inside async Functions

```python
# Error!
# result = await fetch_data()  # SyntaxError

# Correct usage
async def main():
    result = await fetch_data()
```

---

## 3. asyncio Event Loop

### Basic Execution

```python
import asyncio

async def main():
    print("Main coroutine")

# Python 3.7+
asyncio.run(main())

# Or manually
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
```

### Getting Current Loop

```python
async def show_loop():
    loop = asyncio.get_running_loop()
    print(f"Current loop: {loop}")

asyncio.run(show_loop())
```

---

## 4. Creating Tasks

### asyncio.create_task()

Wraps a coroutine as a Task to schedule concurrent execution.

```python
async def task(name, seconds):
    print(f"{name} starting")
    await asyncio.sleep(seconds)
    print(f"{name} done")
    return name

async def main():
    # Create tasks (scheduled immediately)
    task1 = asyncio.create_task(task("A", 2))
    task2 = asyncio.create_task(task("B", 1))

    # Can do other work
    print("Tasks created")

    # Wait for results
    result1 = await task1
    result2 = await task2

    print(f"Results: {result1}, {result2}")

asyncio.run(main())
```

Output:
```
Tasks created
A starting
B starting
B done
A done
Results: A, B
```

---

## 5. Concurrent Execution

### asyncio.gather()

Run multiple coroutines concurrently and wait for all results.

```python
async def fetch(url, delay):
    await asyncio.sleep(delay)
    return f"{url} data"

async def main():
    results = await asyncio.gather(
        fetch("url1", 1),
        fetch("url2", 2),
        fetch("url3", 1),
    )
    print(results)  # ['url1 data', 'url2 data', 'url3 data']

asyncio.run(main())
```

### return_exceptions=True

Continue other tasks even if exceptions occur.

```python
async def might_fail(n):
    if n == 2:
        raise ValueError("Error!")
    await asyncio.sleep(1)
    return n

async def main():
    results = await asyncio.gather(
        might_fail(1),
        might_fail(2),
        might_fail(3),
        return_exceptions=True
    )
    print(results)  # [1, ValueError('Error!'), 3]

asyncio.run(main())
```

### asyncio.wait()

Wait for a set of tasks with finer control.

```python
async def main():
    tasks = [
        asyncio.create_task(fetch("url1", 2)),
        asyncio.create_task(fetch("url2", 1)),
        asyncio.create_task(fetch("url3", 3)),
    ]

    # Return when first completes
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    print(f"Done: {len(done)}, Pending: {len(pending)}")

    # Cancel remaining
    for task in pending:
        task.cancel()
```

### asyncio.as_completed()

Receive results in completion order.

```python
async def main():
    tasks = [
        fetch("url1", 3),
        fetch("url2", 1),
        fetch("url3", 2),
    ]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result)  # Completion order: url2, url3, url1

asyncio.run(main())
```

---

## 6. Timeouts

### asyncio.wait_for()

```python
async def slow_operation():
    await asyncio.sleep(10)
    return "Done"

async def main():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("Timeout!")

asyncio.run(main())
```

### asyncio.timeout() (Python 3.11+)

```python
async def main():
    async with asyncio.timeout(2.0):
        await slow_operation()
```

---

## 7. Async Context Managers

Use `async with`.

```python
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)

async def main():
    async with AsyncResource() as resource:
        print("Using resource")

asyncio.run(main())
```

### contextlib Version

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_resource():
    print("Acquiring")
    yield "resource"
    print("Releasing")

async def main():
    async with async_resource() as r:
        print(f"Using: {r}")
```

---

## 8. Async Iterators

Use `async for`.

```python
class AsyncRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __aiter__(self):
        self.current = self.start
        return self

    async def __anext__(self):
        if self.current >= self.end:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)  # Async operation
        value = self.current
        self.current += 1
        return value

async def main():
    async for num in AsyncRange(0, 5):
        print(num)

asyncio.run(main())
```

### Async Generators

```python
async def async_range(start, end):
    for i in range(start, end):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for num in async_range(0, 5):
        print(num)
```

---

## 9. Practical Example: HTTP Requests

### Using aiohttp

```python
import aiohttp
import asyncio

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "https://example.com",
        "https://httpbin.org/get",
        "https://api.github.com",
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

        for url, result in zip(urls, results):
            print(f"{url}: {len(result)} bytes")

asyncio.run(main())
```

### Async File I/O (aiofiles)

```python
import aiofiles
import asyncio

async def read_file(path):
    async with aiofiles.open(path, 'r') as f:
        return await f.read()

async def write_file(path, content):
    async with aiofiles.open(path, 'w') as f:
        await f.write(content)

async def main():
    await write_file("test.txt", "Hello, Async!")
    content = await read_file("test.txt")
    print(content)

asyncio.run(main())
```

---

## 10. Mixing with Synchronous Code

### run_in_executor()

Run synchronous functions asynchronously.

```python
import asyncio
import time

def blocking_io():
    """Synchronous I/O operation"""
    time.sleep(2)
    return "Result"

async def main():
    loop = asyncio.get_running_loop()

    # Run in thread pool
    result = await loop.run_in_executor(
        None,  # Default ThreadPoolExecutor
        blocking_io
    )
    print(result)

asyncio.run(main())
```

### to_thread() (Python 3.9+)

```python
async def main():
    result = await asyncio.to_thread(blocking_io)
    print(result)
```

### Calling Async from Sync

```python
async def async_func():
    await asyncio.sleep(1)
    return "Result"

# Call from synchronous context
result = asyncio.run(async_func())
print(result)
```

---

## 11. Semaphores and Locks

### Semaphore (Limit Concurrent Execution)

```python
async def limited_task(sem, n):
    async with sem:
        print(f"Task {n} starting")
        await asyncio.sleep(1)
        print(f"Task {n} done")

async def main():
    sem = asyncio.Semaphore(3)  # Max 3 concurrent

    tasks = [limited_task(sem, i) for i in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

### Lock

```python
async def worker(lock, name):
    async with lock:
        print(f"{name} acquired")
        await asyncio.sleep(1)
        print(f"{name} released")

async def main():
    lock = asyncio.Lock()

    await asyncio.gather(
        worker(lock, "A"),
        worker(lock, "B"),
        worker(lock, "C"),
    )

asyncio.run(main())
```

---

## 12. Error Handling

### Task Exception Handling

```python
async def risky_task():
    await asyncio.sleep(1)
    raise ValueError("Error!")

async def main():
    task = asyncio.create_task(risky_task())

    try:
        await task
    except ValueError as e:
        print(f"Exception caught: {e}")

asyncio.run(main())
```

### Multiple Task Exceptions

```python
async def main():
    tasks = [
        asyncio.create_task(task1()),
        asyncio.create_task(task2()),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(f"Success: {result}")
```

---

## 13. Summary

| Concept | Description |
|---------|-------------|
| `async def` | Define coroutine |
| `await` | Wait for coroutine execution |
| `asyncio.run()` | Run event loop |
| `asyncio.create_task()` | Create task |
| `asyncio.gather()` | Run multiple coroutines concurrently |
| `asyncio.wait()` | Fine-grained task management |
| `async with` | Async context manager |
| `async for` | Async iteration |
| `Semaphore` | Limit concurrent execution |

---

## 14. Practice Problems

### Exercise 1: Web Crawler

Write an async crawler that fetches multiple URLs concurrently.

### Exercise 2: Concurrent File Processing

Write a function that reads and processes multiple files concurrently.

### Exercise 3: Rate Limiter

Write an async function that limits requests per second.

---

## Next Steps

Check out [09_Functional_Programming.md](./09_Functional_Programming.md) to learn about functional programming!
