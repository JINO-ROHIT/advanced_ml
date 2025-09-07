from abc import ABC, abstractmethod

# Strategy Interface
class PaymentStrategy(ABC):
    @abstractmethod
    def process_payment(self, amount, details):
        pass

class CreditCardStrategy(PaymentStrategy):
    def process_payment(self, amount, details):
        print(f"Processing ${amount} via Credit Card")
        print(f"Charging card: {details['card_number']}")
        print("Contacting bank...")
        fee = amount * 0.03
        print(f"Transaction fee: ${fee}")
        return True

class PayPalStrategy(PaymentStrategy):
    def process_payment(self, amount, details):
        print(f"Processing ${amount} via PayPal")
        print(f"PayPal email: {details['email']}")
        print("Redirecting to PayPal...")
        fee = amount * 0.025
        print(f"Transaction fee: ${fee}")
        return True

class CryptoStrategy(PaymentStrategy):
    def process_payment(self, amount, details):
        print(f"Processing ${amount} via Cryptocurrency")
        print(f"Wallet address: {details['wallet']}")
        print("Broadcasting to blockchain...")
        fee = amount * 0.01
        print(f"Network fee: ${fee}")
        return True


# context class
class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: PaymentStrategy):
        self.strategy = strategy
    
    def process_payment(self, amount, details):
        return self.strategy.process_payment(amount, details)


# create processor with different strategies
processor = PaymentProcessor(CreditCardStrategy())
processor.process_payment(100, {"card_number": "1234"})

print("\n--- Switching strategy at runtime ---")
processor.set_strategy(PayPalStrategy())
processor.process_payment(75, {"email": "user@example.com"})

print("\n--- Multiple processors with different strategies ---")
credit_processor = PaymentProcessor(CreditCardStrategy())
crypto_processor = PaymentProcessor(CryptoStrategy())

credit_processor.process_payment(50, {"card_number": "9999"})
crypto_processor.process_payment(50, {"wallet": "1A2B3C..."})