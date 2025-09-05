### Pillars of OOPs

1. Encapsulation - "reduce shared state"

Encapsulation is about hiding and protecting internal data so that it can’t be freely messed with.

```
class BankAccount:
    def __init__(self, balance):
        self._balance = balance   # private

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount

    def get_balance(self):
        return self._balance


account = BankAccount(100)
account.deposit(50)
account.withdraw(30)
print(account.get_balance())  
```
- "balance" has no direct access. you can access only via methods.

1. Abstraction - hide unnecessary details and show the users only what matters.

```
from abc import ABC, abstractmethod

# Abstract class: defines WHAT to do, not HOW
class PaymentProcessor(ABC):
    @abstractmethod
    def pay(self, amount):
        pass


class CreditCardProcessor(PaymentProcessor):
    def pay(self, amount):
        print(f"Processing credit card payment of {amount}...")


class PayPalProcessor(PaymentProcessor):
    def pay(self, amount):
        print(f"Processing PayPal payment of {amount}...")


# Client code only sees the abstraction (PaymentProcessor), not implementation
def checkout(processor: PaymentProcessor, amount: int):
    processor.pay(amount)


checkout(CreditCardProcessor(), 100)
checkout(PayPalProcessor(), 200)
```

- users dont care for how the payment is handled.
- they reduce a lot of complexity.

3. Inheritance - easy

4. Polymorphism - same interfaces, but different implementations.

```
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Cow:
    def speak(self):
        return "Moo!"

# Polymorphism in action
def animal_sound(animal):
    print(animal.speak())

animal_sound(Dog())   # Woof!
animal_sound(Cat())   # Meow!
animal_sound(Cow())   # Moo!
```

- the animal_sound() dont care about what class it gets, as long as it has a speak(), it works,