"""
When to use threads --> for I/O bound activities like reading a file, writing to a database etc.
"""


import threading

def hello_world():
    print(f"hello from thread (child) id: {threading.get_native_id()}")

print(f"hello from thread (main) id: {threading.get_native_id()}")

thread = threading.Thread(target = hello_world) # create a thread and ask it to run hello_world function
thread.start() # this starts the thread