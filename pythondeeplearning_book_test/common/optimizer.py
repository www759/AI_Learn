class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def updata(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

