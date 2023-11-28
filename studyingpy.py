class Account:
    def __init__(self, accountNumber = 1, balance = 0):
        self.balance = balance
        self.accountNumber = accountNumber


    def withdrawl(self, amount):
        if amount > self.balance:
            return "nope "
        
        else:
            self.balance -= amount

        print(f"Withdrew {amount}   New balance = {self.balance}")
        
    def deposit(self, amount):
        if amount < 0:
            print("really?")
        else:
            self.balance += amount
            print(f"deposited == {amount}    New balance = {self.balance}")


newAccount = Account()
newAccount.deposit(400)
newAccount.withdrawl(350)
