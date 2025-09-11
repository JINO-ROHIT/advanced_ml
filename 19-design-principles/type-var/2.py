from typing import TypeVar, Generic, List, Optional

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> Optional[T]:
        return self._items.pop() if self._items else None
    
    def peek(self) -> Optional[T]:
        return self._items[-1] if self._items else None

int_stack = Stack[int]()        # T becomes int
int_stack.push(42)              # works
int_stack.push("hello")         # error never at runtime but static checking

print(int_stack._items)

str_stack = Stack[str]()        # T becomes str  
str_stack.push("hello")         # works
str_stack.push(42)              # same