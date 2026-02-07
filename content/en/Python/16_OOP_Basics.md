# Object-Oriented Programming Basics

> **Note**: This lesson is for prerequisite knowledge review. If you lack OOP fundamentals before starting advanced lessons (decorators, metaclasses, etc.), study this content first.

## Learning Objectives
- Understand the concepts of classes and objects (instances)
- Utilize constructors, instance/class variables, and methods
- Grasp the principles of inheritance, polymorphism, and encapsulation
- Use basic special methods (magic methods)

---

## 1. Classes and Objects

### 1.1 Basic Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                Object-Oriented Programming (OOP)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Class:                                                         │
│  - Blueprint (design) for objects                               │
│  - Defines attributes (variables) and behaviors (methods)       │
│                                                                 │
│  Object / Instance:                                             │
│  - Concrete entity created from a class                         │
│  - Each object has independent state (data)                     │
│                                                                 │
│  Example:                                                       │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │ class Dog   │ ──────▶ │ my_dog      │  # Object 1           │
│  │  name       │         │ name="Max"  │                       │
│  │  age        │         │ age=3       │                       │
│  │  bark()     │         └─────────────┘                       │
│  │  eat()      │                                                │
│  └─────────────┘ ──────▶ ┌─────────────┐                       │
│                          │ your_dog    │  # Object 2           │
│                          │ name="Bella"│                       │
│                          │ age=5       │                       │
│                          └─────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Class Definition

```python
class Dog:
    """Dog class"""

    # Class variable (shared by all instances)
    species = "Canis familiaris"

    # Constructor (initialization method)
    def __init__(self, name, age):
        """
        Args:
            name: Dog's name
            age: Dog's age
        """
        # Instance variables (independent for each object)
        self.name = name
        self.age = age

    # Instance method
    def bark(self):
        """Bark"""
        print(f"{self.name} says: Woof!")

    def describe(self):
        """Description"""
        return f"{self.name} is {self.age} years old"

    def birthday(self):
        """Birthday"""
        self.age += 1
        print(f"Happy birthday, {self.name}! Now {self.age} years old.")


# Object creation
my_dog = Dog("Max", 3)
your_dog = Dog("Bella", 5)

# Attribute access
print(my_dog.name)     # Max
print(my_dog.species)  # Canis familiaris

# Method calls
my_dog.bark()           # Max says: Woof!
print(my_dog.describe()) # Max is 3 years old
my_dog.birthday()        # Happy birthday, Max! Now 4 years old.

# Class variable vs Instance variable
print(Dog.species)       # Access via class
print(my_dog.species)    # Access via instance (same value)
```

### 1.3 Meaning of self

```python
class Circle:
    def __init__(self, radius):
        # self = refers to the current instance being created
        self.radius = radius

    def area(self):
        # self.radius = this instance's radius
        return 3.14159 * self.radius ** 2

    def circumference(self):
        return 2 * 3.14159 * self.radius


# When creating objects
c1 = Circle(5)   # self = c1
c2 = Circle(10)  # self = c2

# When calling methods
print(c1.area())  # self = c1, self.radius = 5
print(c2.area())  # self = c2, self.radius = 10

# Actually works like this:
# Circle.area(c1) → self = c1
```

---

## 2. Class/Instance Variables and Methods

### 2.1 Variable Types

```python
class Counter:
    # Class variable: shared by all instances
    total_count = 0

    def __init__(self, name):
        # Instance variable: independent for each object
        self.name = name
        self.count = 0

        # Modify class variable
        Counter.total_count += 1

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count


c1 = Counter("Counter 1")
c2 = Counter("Counter 2")

c1.increment()
c1.increment()
c2.increment()

print(c1.count)  # 2 (instance variable)
print(c2.count)  # 1 (instance variable)
print(Counter.total_count)  # 2 (class variable)

# Warning: assigning to class variable via instance creates instance variable
c1.total_count = 100  # creates instance variable for c1 only
print(c1.total_count)       # 100 (instance)
print(Counter.total_count)  # 2 (class variable unchanged)
```

### 2.2 Method Types

```python
class MyClass:
    class_var = 0

    def __init__(self, value):
        self.instance_var = value

    # Instance method: uses self, accesses instance data
    def instance_method(self):
        return f"Instance: {self.instance_var}"

    # Class method: uses cls, accesses class data
    @classmethod
    def class_method(cls):
        cls.class_var += 1
        return f"Class var: {cls.class_var}"

    # Static method: no self or cls, independent function
    @staticmethod
    def static_method(x, y):
        return x + y


obj = MyClass(42)

# Instance method
print(obj.instance_method())  # Instance: 42

# Class method (callable via class or instance)
print(MyClass.class_method())  # Class var: 1
print(obj.class_method())      # Class var: 2

# Static method (callable via class or instance)
print(MyClass.static_method(3, 4))  # 7
print(obj.static_method(3, 4))      # 7
```

### 2.3 Factory Method Pattern

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

    # Factory method: create objects from various formats
    @classmethod
    def from_string(cls, date_string):
        """Create from 'YYYY-MM-DD' format"""
        year, month, day = map(int, date_string.split("-"))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        """Create from today's date"""
        import datetime
        t = datetime.date.today()
        return cls(t.year, t.month, t.day)


# Create objects in various ways
d1 = Date(2024, 1, 15)
d2 = Date.from_string("2024-06-20")
d3 = Date.today()

print(d1)  # Date(2024, 1, 15)
print(d2)  # Date(2024, 6, 20)
print(d3)  # Date(current date)
```

---

## 3. Inheritance

### 3.1 Basic Inheritance

```python
# Parent class (superclass, base class)
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement")

    def describe(self):
        return f"I am {self.name}"


# Child class (subclass, derived class)
class Dog(Animal):
    def __init__(self, name, breed):
        # Initialize parent class
        super().__init__(name)
        self.breed = breed

    def speak(self):
        return f"{self.name} says Woof!"

    def fetch(self):
        return f"{self.name} is fetching the ball"


class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

    def scratch(self):
        return f"{self.name} is scratching"


# Usage
dog = Dog("Max", "Golden Retriever")
cat = Cat("Whiskers")

print(dog.describe())  # I am Max (parent method)
print(dog.speak())     # Max says Woof! (overridden)
print(dog.fetch())     # Max is fetching the ball (child-specific)
print(dog.breed)       # Golden Retriever

print(cat.speak())     # Whiskers says Meow!

# Check inheritance relationships
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True
print(issubclass(Dog, Animal))  # True
```

### 3.2 Method Overriding

```python
class Vehicle:
    def __init__(self, brand):
        self.brand = brand

    def info(self):
        return f"Brand: {self.brand}"

    def move(self):
        return "Moving..."


class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model

    # Method overriding (redefinition)
    def info(self):
        # Call parent method + extend
        parent_info = super().info()
        return f"{parent_info}, Model: {self.model}"

    def move(self):
        # Completely new implementation
        return "Driving on the road"

    def honk(self):
        return "Beep beep!"


class Motorcycle(Vehicle):
    def move(self):
        return "Riding on two wheels"


car = Car("Toyota", "Camry")
print(car.info())  # Brand: Toyota, Model: Camry
print(car.move())  # Driving on the road

moto = Motorcycle("Harley")
print(moto.move())  # Riding on two wheels
```

### 3.3 Multiple Inheritance

```python
class Flyable:
    def fly(self):
        return "Flying in the sky"


class Swimmable:
    def swim(self):
        return "Swimming in the water"


class Duck(Animal, Flyable, Swimmable):
    def speak(self):
        return f"{self.name} says Quack!"


duck = Duck("Donald")
print(duck.describe())  # I am Donald (Animal)
print(duck.fly())       # Flying in the sky (Flyable)
print(duck.swim())      # Swimming in the water (Swimmable)
print(duck.speak())     # Donald says Quack!

# Check MRO (Method Resolution Order)
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Animal'>, <class 'Flyable'>, <class 'Swimmable'>, <class 'object'>)
```

---

## 4. Encapsulation

### 4.1 Access Control

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner          # public
        self._balance = balance     # protected (convention)
        self.__pin = "1234"         # private (name mangling)

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return True
        return False

    def withdraw(self, amount, pin):
        if pin != self.__pin:
            raise ValueError("Invalid PIN")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        return amount

    def get_balance(self):
        return self._balance


account = BankAccount("Alice", 1000)

# public access
print(account.owner)  # Alice

# protected access (possible but not recommended)
print(account._balance)  # 1000

# private access (not directly accessible)
# print(account.__pin)  # AttributeError

# Access via name mangling (not recommended)
print(account._BankAccount__pin)  # 1234

# Safe access through methods
account.deposit(500)
print(account.get_balance())  # 1500
```

### 4.2 Property

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius

    # getter
    @property
    def celsius(self):
        return self._celsius

    # setter
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    # Computed property
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9


temp = Temperature()

# Use property (like an attribute)
temp.celsius = 25
print(temp.celsius)      # 25
print(temp.fahrenheit)   # 77.0

temp.fahrenheit = 100
print(temp.celsius)      # 37.777...

# Validation works
# temp.celsius = -300  # ValueError
```

---

## 5. Polymorphism

### 5.1 Method Polymorphism

```python
class Shape:
    def area(self):
        raise NotImplementedError


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2


class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height


# Polymorphism: same method name, different behavior
def print_area(shape: Shape):
    print(f"Area: {shape.area()}")


shapes = [
    Rectangle(4, 5),
    Circle(3),
    Triangle(6, 8)
]

for shape in shapes:
    print_area(shape)

# Output:
# Area: 20
# Area: 28.27431
# Area: 24.0
```

### 5.2 Duck Typing

```python
# "If it walks like a duck and quacks like a duck, it's a duck"
# Python focuses on what methods/attributes an object has rather than its type

class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Robot:
    def speak(self):
        return "Beep boop!"

# No inheritance relationship needed - same method works the same
def animal_sound(animal):
    print(animal.speak())

# All have speak() method so they work
animal_sound(Dog())    # Woof!
animal_sound(Cat())    # Meow!
animal_sound(Robot())  # Beep boop!
```

---

## 6. Special Methods (Magic Methods)

### 6.1 Basic Special Methods

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # String representation (for users)
    def __str__(self):
        return f"Point({self.x}, {self.y})"

    # String representation (for developers, debugging)
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    # Equality comparison
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False

    # Hash (can be used as dict key, set element)
    def __hash__(self):
        return hash((self.x, self.y))

    # Length
    def __len__(self):
        return 2  # two coordinates: x, y

    # Boolean conversion
    def __bool__(self):
        return self.x != 0 or self.y != 0


p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(0, 0)

print(str(p1))   # Point(3, 4)
print(repr(p1))  # Point(x=3, y=4)
print(p1 == p2)  # True
print(len(p1))   # 2
print(bool(p1))  # True
print(bool(p3))  # False
```

### 6.2 Operator Overloading

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    # Addition: v1 + v2
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    # Subtraction: v1 - v2
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    # Multiplication (scalar): v * 3
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    # Reverse multiplication: 3 * v
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    # Dot product: v1 @ v2
    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y

    # Negation: -v
    def __neg__(self):
        return Vector(-self.x, -self.y)

    # Absolute value (magnitude)
    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)     # Vector(4, 6)
print(v1 - v2)     # Vector(2, 2)
print(v1 * 2)      # Vector(6, 8)
print(3 * v1)      # Vector(9, 12)
print(v1 @ v2)     # 11 (dot product)
print(-v1)         # Vector(-3, -4)
print(abs(v1))     # 5.0
```

### 6.3 Container Protocol

```python
class Deck:
    """Card deck class"""

    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [f"{rank} of {suit}" for suit in suits for rank in ranks]

    # Length
    def __len__(self):
        return len(self.cards)

    # Indexing: deck[0]
    def __getitem__(self, index):
        return self.cards[index]

    # Assignment: deck[0] = "Joker"
    def __setitem__(self, index, value):
        self.cards[index] = value

    # Deletion: del deck[0]
    def __delitem__(self, index):
        del self.cards[index]

    # Membership: "Ace of Spades" in deck
    def __contains__(self, item):
        return item in self.cards

    # Iteration
    def __iter__(self):
        return iter(self.cards)


deck = Deck()

print(len(deck))          # 52
print(deck[0])            # 2 of Hearts
print(deck[-1])           # A of Spades
print("A of Spades" in deck)  # True

# Slicing automatically supported
print(deck[0:3])  # ['2 of Hearts', '3 of Hearts', '4 of Hearts']

# Iteration
for card in deck[:5]:
    print(card)
```

---

## 7. Abstract Classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class"""

    @abstractmethod
    def area(self):
        """Calculate area (must be implemented)"""
        pass

    @abstractmethod
    def perimeter(self):
        """Calculate perimeter (must be implemented)"""
        pass

    def describe(self):
        """Regular method (inherited)"""
        return f"Area: {self.area()}, Perimeter: {self.perimeter()}"


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14159 * self.radius


# Abstract class cannot be instantiated
# shape = Shape()  # TypeError

rect = Rectangle(4, 5)
print(rect.describe())  # Area: 20, Perimeter: 18

circle = Circle(3)
print(circle.describe())  # Area: 28.27..., Perimeter: 18.84...
```

---

## Summary

### Core OOP Concepts

| Concept | Description |
|---------|-------------|
| Class | Blueprint for objects |
| Object/Instance | Concrete entity created from a class |
| `__init__` | Constructor, initialization method |
| `self` | Reference to current instance |
| Inheritance | Inheriting attributes/methods from parent class |
| Overriding | Redefining parent methods |
| Encapsulation | Data protection, access control |
| Polymorphism | Same interface, different behavior |
| Abstract Class | Base class with incomplete methods |

### Access Control Conventions

| Format | Meaning | Example |
|--------|---------|---------|
| `name` | public | Freely accessible |
| `_name` | protected | Internal use recommended |
| `__name` | private | Name mangling applied |

### Next Steps

After completing OOP basics, proceed to:
- [02_Decorators.md](./02_Decorators.md): Decorators
- [06_Metaclasses.md](./06_Metaclasses.md): Metaclasses
- [07_Descriptors.md](./07_Descriptors.md): Descriptors

---

## References

- [Python OOP Official Documentation](https://docs.python.org/3/tutorial/classes.html)
- [Real Python - OOP](https://realpython.com/python3-object-oriented-programming/)
- [Data Model Documentation](https://docs.python.org/3/reference/datamodel.html)
