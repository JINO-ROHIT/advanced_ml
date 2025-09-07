'''
one thing you need to know about coroutines vs tasks is,
- making a task will schedule in the event loop even before await, so whenever it has chance to run, it will get started and then switch to another task on pauses.
- if you await coroutines alone, it also awaits but it wont have the other task scheduled, so it has to wait till the task1 is done and then scheudle and complete task2.
'''
import asyncio
import time


# async def fetch_data(param):
#     print(f"Do something with {param}...")
#     await asyncio.sleep(param)
#     print(f"Done with {param}")
#     return f"Result of {param}"

# async def main():
#     task1 = asyncio.create_task(fetch_data(1))
#     task2 = asyncio.create_task(fetch_data(2))
#     result1 = await task1
#     print("task 1 done")
#     result2 = await task2
#     print("task 2 done")
#     return [result1, result2]

# t1 = time.perf_counter()
# results = asyncio.run(main())
# print(results)
# t2 = time.perf_counter()

# print(f"Finished in {t2 - t1:.2f} seconds")


async def fetch_data(param):
    print(f"Do something with {param}...")
    await asyncio.sleep(param)
    print(f"Done with {param}")
    return f"Result of {param}"

async def main():
    result1 = await fetch_data(1)
    print("coroutine 1 done")
    result2 = await fetch_data(2)
    print("coroutine 2 done")
    return [result1, result2]

t1 = time.perf_counter()
results = asyncio.run(main())
print(results)
t2 = time.perf_counter()

print(f"Finished in {t2 - t1:.2f} seconds")