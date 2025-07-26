"""from the previous example, if you wanted to make the main thread wait till the child thread finishes you can use join()"""

import threading
import time

def hello_world():
    time.sleep(3)
    print(f"hello from thread (child) id: {threading.get_native_id()}")

print(f"hello from thread (main) id: {threading.get_native_id()}")

thread = threading.Thread(target = hello_world, daemon = True)
thread.start()

print("main thread is about to finish. daemon thread will be killed if not done.")
print("sike")
thread.join()