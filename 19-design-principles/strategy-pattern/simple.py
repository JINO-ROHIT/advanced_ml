class BadPaymentProcessor:
    def process_payment(self, amount, payment_type, details):

        if payment_type == "credit_card":
            print(f"Processing ${amount} via Credit Card")
            print(f"Charging card: {details['card_number']}")
            print("Contacting bank...")
            fee = amount * 0.03 
            print(f"Transaction fee: ${fee}")
            return True
            
        elif payment_type == "paypal":
            print(f"Processing ${amount} via PayPal")
            print(f"PayPal email: {details['email']}")
            print("Redirecting to PayPal...")
            fee = amount * 0.025  
            print(f"Transaction fee: ${fee}")
            return True
            
        elif payment_type == "crypto":
            print(f"Processing ${amount} via Cryptocurrency")
            print(f"Wallet address: {details['wallet']}")
            print("Broadcasting to blockchain...")
            fee = amount * 0.01 
            print(f"Network fee: ${fee}")
            return True
        else:
            print("Unsupported payment method!")
            return False
    
    # adding new payment methods requires modifying this class!

print("=== BEFORE: Without Strategy Pattern ===")
processor = BadPaymentProcessor()
processor.process_payment(100, "credit_card", {"card_number": "1234"})
print()