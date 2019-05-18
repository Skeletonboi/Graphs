class stack:
    def __init__(self):
        self.store = []

    def push(self, x):
        self.store = self.store + [x]

    def pop(self):
        if len(self.store) == 0:
            return False
        elif len(self.store) == 1:
            val = self.store[0]
            self.store = []
        else:
            val = self.store[-1]
            self.store = self.store[:-1]
        return val
