from typing import TypeVar

T = TypeVar('T')  # T can be any type

def identity(value: T) -> T:
    return value

x = identity(42)        # T becomes int, returns int
y = identity("hello")   # T becomes str, returns str  
z = identity([1,2,3])   # T becomes list[int], returns list[int]

print(type(x), type(y), type(z))



## without typevar

from typing import Any

def identity(value: Any) -> Any:
    return value

x = identity(42)  # Type checker thinks this is Any, not int
print(type(x))