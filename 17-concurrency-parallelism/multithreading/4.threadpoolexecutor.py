"""This section will show you how to run a group of threads"""

import concurrent.futures
import threading

def hello_world(dummy_id):
    print(f"{dummy_id} : hello from thread (child) id: {threading.get_native_id()}")

with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as manager:
    manager.map(hello_world, range(10)) # run hello_world 10 times among 3 workers