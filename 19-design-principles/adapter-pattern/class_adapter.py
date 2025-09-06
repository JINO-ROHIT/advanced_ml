class PaymentProcessor:
    def pay(self, amount: float) -> None:
        raise NotImplementedError

class StripeAPI:
    def make_payment(self, amount_in_cents: int) -> None:
        print(f"Stripe: Paid {amount_in_cents/100:.2f} USD")

class PayPalAPI:
    def send_money(self, user_email: str, amount: float) -> None:
        print(f"PayPal: Sent {amount:.2f} USD to {user_email}")



# the StripeAPI and PayPalAPI is incompactible, so we make a adapter

class StripeAdapter(PaymentProcessor): # inherit
    def __init__(self, stripe_api):
        self._stripe = stripe_api
    
    def pay(self, amount: float) -> None:
        cents = int(amount * 100)
        self._stripe.make_payment(cents)

class PayPalAdapter(PaymentProcessor):
    def __init__(self, paypal_api: PayPalAPI, email: str):
        self._paypal = paypal_api
        self._email = email

    def pay(self, amount: float) -> None:
        self._paypal.send_money(self._email, amount)

if __name__ == "__main__":
    stripe = StripeAdapter(StripeAPI())
    paypal = PayPalAdapter(PayPalAPI(), "customer@example.com")

    for processor in [stripe, paypal]:
        processor.pay(50.0)

