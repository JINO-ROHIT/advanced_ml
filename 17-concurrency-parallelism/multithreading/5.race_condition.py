"""This example shows what happens when two threads try to access a shared resource."""

import concurrent.futures
import threading
import time

class Bank:
    def __init__(self):
        self.money = 0
    
    def pay(self, amount):
        current = self.money
        time.sleep(0.001)
        self.money = current + amount
        print(f"You have paid { amount} from thread {threading.get_native_id()}. Current standing : {self.money}")

bank = Bank()

with concurrent.futures.ThreadPoolExecutor(max_workers = 10) as manager:
    manager.map(bank.pay, [10] * 10)

print(f"Final total amount : {bank.money}") # re-run this multiple times and see what happens.