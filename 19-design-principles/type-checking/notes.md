### If TYPE_CHECKING 

The **if TYPE_CHECKING** pattern is a powerful technique for handling imports that are only needed for type annotations not at runtime.

TYPE_CHECKING is a special constant from the typing module that:
- is False during normal runtime execution
- is True when static type checkers (mypy) are analyzing your code