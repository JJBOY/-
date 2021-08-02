import numpy as np


class Linear_Regression(object):
    def __init__(self, feat_dim=1):
        self.W = np.random.randn(feat_dim + 1, 1)
        self.W[-1] *= 0

    def train(self, X, y, epoch=100, lr=0.001):
        """
        :param X: shape:(N,C)
        :param y: shape:(N,1)
        :param epoch: int
        :param lr: learning rate
        """
        X_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        for i in range(epoch):
            y_ = np.dot(X_bias, self.W)  # (N,1)
            loss = np.mean(np.square(y_ - y))
            dW = np.dot(X_bias.T, y_ - y) / X.shape[0]
            self.W[:-1] -= lr * dW[:-1]
            self.W[-1] -= 10 * lr * dW[-1]
            # self.show(X, y)
            if i == int(epoch * 0.5) or i == int(epoch * 0.8):
                lr /= 10

    def predict(self, X):
        X_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return np.dot(X_bias, self.W)

    def show(self, X, y):
        """
        only for X shape=(N,1)
        """
        import matplotlib.pyplot as plt
        plt.scatter(X[:, 0], y)
        plt.plot(X[:, 0], self.predict(X))
        plt.show()


class Logistic_Regression(object):
    def __init__(self, feat_dim=1):
        self.W = np.random.randn(feat_dim + 1, 1)
        self.W[-1] *= 0

    def train(self, X, y, epoch=2000, lr=0.01):
        """
        :param X: shape:(N,C)
        :param y: shape:(N,1)
        :param epoch: int
        :param lr: learning rate
        """
        X_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        for i in range(epoch):
            y_ = self._sigmoid(np.dot(X_bias, self.W))  # (N,1)
            loss = -np.mean(y * np.log(y_) + (1 - y) * np.log(1 - y_))
            dW = np.dot(X_bias.T, y_ - y) / X.shape[0]
            self.W[:-1] -= lr * dW[:-1]
            self.W[-1] -= 10 * lr * dW[-1]
            # self.show(X, y)
            if i == int(epoch * 0.5) or i == int(epoch * 0.8):
                lr /= 10

    def predict(self, X):
        X_bias = np.concatenate([X, np.ones(X.shape[0], 1)], axis=1)
        return self._sigmoid(np.dot(X_bias, self.W))

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def show(self, X, y):
        """
        only for X shape=(N,2)
        """
        import matplotlib.pyplot as plt
        plt.scatter(X[:, 0], X[:, 1], c=y)
        x_range = np.linspace(start=np.min(X[:, 0]), stop=np.max(X[:, 0]), num=100).reshape(100, 1)
        X_bias = np.concatenate([x_range, np.ones((100, 1))], axis=1)
        y_ = -np.dot(X_bias, np.concatenate([self.W[0:1, ], self.W[2:, ]], axis=0)) / self.W[1]
        plt.plot(x_range, y_)
        plt.show()


if __name__ == '__main__':
    # LR = Linear_Regression(1)
    # X = np.arange(20).reshape(20, 1)
    # y = 10 * X + np.ones_like(X) * 7 + np.random.randn(20).reshape(20, 1) * 10
    # LR.train(X, y)
    # LR.show(X, y)

    LogR = Logistic_Regression(2)
    X1 = np.concatenate([np.random.randn(50).reshape(50, 1) + 3, np.random.randn(50).reshape(50, 1) + 10], axis=1)
    X2 = np.concatenate([np.random.randn(50).reshape(50, 1) + 6, np.random.randn(50).reshape(50, 1) + 7], axis=1)
    X = np.concatenate([X1, X2], axis=0)
    y = np.array([1] * 50 + [0] * 50).reshape(100, 1)
    LogR.train(X, y)
    LogR.show(X, y)
