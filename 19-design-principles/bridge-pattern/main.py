from db_backend import SQLiteDB, PostgresDB
from operation import Repository

if __name__ == "__main__":
    sqlite_repo = Repository(SQLiteDB())
    postgres_repo = Repository(PostgresDB())

    sqlite_repo._db.connect()
    sqlite_repo.insert("users", {"name": "Alice", "email": "alice@example.com"})
    sqlite_repo.select("users")

    print("---")

    postgres_repo._db.connect()
    postgres_repo.insert("users", {"name": "Bob", "email": "bob@example.com"})
    postgres_repo.select("users", "name='Bob'")
