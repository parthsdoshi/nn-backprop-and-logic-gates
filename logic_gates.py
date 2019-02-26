import torch
from neural_network import NeuralNetwork


class AND():
    def __init__(self):
        self.model = NeuralNetwork([2, 1])

        self.inputs = torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])
        self.labels = torch.tensor([
            [0.],
            [0.],
            [0.],
            [1.]
        ])
    
    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        inputs = self.inputs
        labels = self.labels

        it = 0
        mse = float('inf')
        while (mse >= 0.01 and it < 100000):
            y = self.model.forward(inputs)
            self.model.backward(labels)
            self.model.updateParams()
            it += 1

            n_mse = self.model.mse(inputs, labels)
            if n_mse > mse:
                break
            mse = n_mse

        # mse = self._mse()
        # print(f"AND MSE: {mse}")

    def forward(self, x1, x2):
        x = torch.tensor([[x1, x2]], dtype=torch.float)
        return self.model.forward(x).item() >= 0.5

    def _mse(self):
        return self.model.mse(self.inputs, self.labels)


class OR():
    def __init__(self):
        self.model = NeuralNetwork([2, 1])
        self.inputs = torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])
        self.labels = torch.tensor([
            [0.],
            [1.],
            [1.],
            [1.]
        ])

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        inputs = self.inputs
        labels = self.labels

        it = 0
        mse = float('inf')
        while (mse >= 0.01 and it < 100000):
            y = self.model.forward(inputs)
            self.model.backward(labels)
            self.model.updateParams()
            it += 1

            n_mse = self.model.mse(inputs, labels)
            if n_mse > mse:
                break
            mse = n_mse

        # mse = self._mse()
        # print(f"OR MSE: {mse}")

    def forward(self, x1, x2):
        x = torch.tensor([[x1, x2]], dtype=torch.float)
        return self.model.forward(x).item() >= 0.5

    def _mse(self):
        return self.model.mse(self.inputs, self.labels)


class NOT():
    def __init__(self):
        self.model = NeuralNetwork([1, 1])

        self.inputs = torch.tensor([
            [0.],
            [1.]
        ])
        self.labels = torch.tensor([
            [1.],
            [0.]
        ])

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        inputs = self.inputs
        labels = self.labels

        it = 0
        mse = float('inf')
        while (mse >= 0.01 and it < 100000):
            y = self.model.forward(inputs)
            self.model.backward(labels)
            self.model.updateParams()
            it += 1

            n_mse = self.model.mse(inputs, labels)
            if n_mse > mse:
                break
            mse = n_mse

        # mse = self._mse()
        # print(f"NOT MSE: {mse}")

    def forward(self, x):
        x = torch.tensor([[x]], dtype=torch.float)
        return self.model.forward(x).item() >= 0.5
    
    def _mse(self):
        return self.model.mse(self.inputs, self.labels)


class XOR():
    def __init__(self):
        self.model = NeuralNetwork([2, 2, 1])

        self.inputs = torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.]
        ])
        self.labels = torch.tensor([
            [0.],
            [1.],
            [1.],
            [0.]
        ])

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        inputs = self.inputs
        labels = self.labels

        iter = 0
        # while loop because sometimes xor doesn't converge
        while (self._mse() >= 0.01 and iter < 100):
            self.__init__()
            iter += 1

            it = 0
            mse = float('inf')
            while (mse >= 0.01 and it < 100000):
                y = self.model.forward(inputs)
                self.model.backward(labels)

                # xor needs a super small learning rate
                self.model.updateParams(.5)

                it += 1

                n_mse = self.model.mse(inputs, labels)
                if (n_mse > mse):
                    break
                mse = n_mse

        # mse = self._mse()
        # print(f"XOR MSE: {mse}")

    def forward(self, x1, x2):
        x = torch.tensor([[x1, x2]], dtype=torch.float)
        return self.model.forward(x).item() >= 0.5
    
    def _mse(self):
        return self.model.mse(self.inputs, self.labels)