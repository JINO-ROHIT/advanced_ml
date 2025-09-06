from db_backend import DBImplementation

# the abstraction 

class Repository:
    def __init__(self, db: DBImplementation):
        self._db = db   # bridge: composition

    def insert(self, table: str, values: dict[str, str]) -> None:
        columns = ", ".join(values.keys())
        vals = ", ".join(f"'{v}'" for v in values.values())
        query = f"INSERT INTO {table} ({columns}) VALUES ({vals});"
        self._db.execute(query)

    def select(self, table: str, where: str = "1=1") -> None:
        query = f"SELECT * FROM {table} WHERE {where};"
        self._db.execute(query)
