"""
Main thread - When you run a python script, this thread starts executing your python script code.
Daemon thread - This thread is a simply a background thread that terminates when main thread is done executing.
For example to do garbage collection, you can use daemon thread, run it and forget about it. As soon as main thread is done, the daemon thread stops executing.
"""

import threading
import time

def hello_world():
    time.sleep(5)
    print(f"hello from thread (child) id: {threading.get_native_id()}")

print(f"hello from thread (main) id: {threading.get_native_id()}")

thread = threading.Thread(target = hello_world, daemon = True)
thread.start()

print("main thread is about to finish. daemon thread will be killed if not done.")
# set deamon to True vs False and observe whats the difference.