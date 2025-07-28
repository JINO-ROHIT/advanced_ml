"""In this tutorial, we will see how to return future from thread pool executor"""

import threading
import concurrent.futures
import os
import time
from random import randint

def return_paths(path):
    time.sleep(randint(1, 5))
    print(f"thread {threading.get_native_id()} is seeing {path}")
    return path

files_paths = []
for root, _, files in os.walk('data'):
    for file in files:
        files_paths.append(os.path.join(root, file).replace('\\', '/')) # windows path

with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as executor:
    print(f"Order of iterables: {files_paths}")
    futures = [executor.submit(return_paths, file_path) for file_path in files_paths]

    # print(futures)

    for future in concurrent.futures.as_completed(futures):
        file_path = future.result()
        print(file_path)