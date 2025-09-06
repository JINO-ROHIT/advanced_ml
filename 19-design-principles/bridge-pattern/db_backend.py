from abc import ABC, abstractmethod

# the implementation

class DBImplementation(ABC):
    @abstractmethod
    def connect(self) -> None: 
        ...
    
    @abstractmethod
    def execute(self, query: str) -> None: 
        ...


class SQLiteDB(DBImplementation):
    def connect(self) -> None:
        print("Connecting to SQLite...")
    
    def execute(self, query: str) -> None:
        print(f"[SQLite] Executing: {query}")


class PostgresDB(DBImplementation):
    def connect(self) -> None:
        print("Connecting to PostgreSQL...")
    
    def execute(self, query: str) -> None:
        print(f"[Postgres] Executing: {query}")
