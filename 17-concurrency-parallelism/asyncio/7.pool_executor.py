'''
now if you have a sync blocking function, you can do things -
1. offload to a thread
2. use a process pool
'''

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor

def fetch_data(param):
    print(f"Do something with {param}...", flush=True)
    time.sleep(param)
    print(f"Done with {param}", flush=True)
    return f"Result of {param}"

# using thread
async def main():
    task1 = asyncio.create_task(asyncio.to_thread(fetch_data, 1))
    task2 = asyncio.create_task(asyncio.to_thread(fetch_data, 2))
    result1 = await task1
    print("Thread 1 fully completed")
    result2 = await task2
    print("Thread 2 fully completed")

# using process
async def main2():

    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor() as executor:
        task1 = loop.run_in_executor(executor, fetch_data, 1)
        task2 = loop.run_in_executor(executor, fetch_data, 2)

        result1 = await task1
        print("Process 1 fully completed")
        result2 = await task2
        print("Process 2 fully completed")


"""
Under Windows, it is important to protect the main loop of code to avoid recursive spawning of
subprocesses when using processpoolexecutor or any other parallel code which spawns new processes.
Basically, all your code which creates new processes must be under if __name__ == '__main__':  
for the same reason you cannot execute it in interpreter.
"""

if __name__ == '__main__':
    t1 = time.perf_counter()
    #results = asyncio.run(main())
    results = asyncio.run(main2())
    t2 = time.perf_counter()

    print(f"Finished in {t2 - t1:.2f} seconds")
