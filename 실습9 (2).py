class BandAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def Deposit(self, amount):
        if amount > 0:
            self.balance = self.balance + amount

    def withdraw(self, amount):
        if self.balance > amount:
            self.balance = self.balance - amount
        else:
            print("출금 불가능")

    def display_balance(self):
        print("현재 잔액:", self.balance)


a1 = BandAccount("하이", 10000)
a1.Deposit(5000)
a1.display_balance()