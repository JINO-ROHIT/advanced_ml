"""In this tutorial, we will see how to return values from thread pool executor"""

import threading
import concurrent.futures
import os
import time

def return_paths(path):
    time.sleep(5)
    print(f"thread {threading.get_native_id()} is seeing {path}")
    return path

files_paths = []
for root, _, files in os.walk('data'):
    for file in files:
        files_paths.append(os.path.join(root, file).replace('\\', '/')) # windows path

with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as executor:
    print(f"Order of iterables: {files_paths}")
    results = executor.map(return_paths, files_paths)

    for result in results:
        print(result)

# look at the order of iterables and the return order of results.
# Results will be returned in the order of the iterables you have, not in the order the task gets completed. 