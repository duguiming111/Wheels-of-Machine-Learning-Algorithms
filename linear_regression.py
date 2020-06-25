# Author: dgm
# Description: 手撸线性回归
# Date: 2020-06-25
import numpy as np
import matplotlib.pyplot as plt


def data_generate():
    """产生数据"""
    x = np.arange(0, 10, 0.2)
    y = 4 * x + 1 + np.random.randn(len(x))

    return x, y


class LinearRegression(object):
    def __init__(self, learning_rate, iters):
        self.W = None
        self.b = None
        self.learning_rate = learning_rate
        self.iters = iters

    def loss(self, x, y):
        num_feature = x.shape[1]
        num_train = x.shape[0]

        h = x.dot(self.W) + self.b
        loss = 0.5 * np.sum(np.square(h - y)) / num_train

        dW = x.T.dot((h - y)) / num_train
        db = np.sum((h - y)) / num_train

        return loss, dW, db

    def fit(self, x, y):
        num_feature = x.shape[1]
        self.W = np.zeros((num_feature, 1))
        self.b = 0
        loss_list = []

        for i in range(self.iters):
            loss, dW, db = self.loss(x, y)
            loss_list.append(loss)
            self.W += -self.learning_rate * dW
            self.b += -self.learning_rate * db

            if i % 500 == 0:
                print('iters = %d,loss = %f' % (i, loss))
        return loss_list


if __name__ == '__main__':
    x, y = data_generate()
    x_train = list()
    y_train = list()
    for item in x:
        tmp = list()
        tmp.append(item)
        x_train.append(tmp)
    for item in y:
        tmp = list()
        tmp.append(item)
        y_train.append(tmp)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    lr = LinearRegression(0.001, 1000)
    loss = lr.fit(x_train, y_train)
    f = x_train.dot(lr.W) + lr.b

    # 画图
    fig = plt.figure()
    plt.subplot(211)
    plt.scatter(x_train, y_train, color='black')
    plt.plot(x_train, f, 'r-')
    plt.xlabel('x')
    plt.xlabel('y')

    plt.subplot(212)
    plt.plot(loss, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('errors')
    plt.show()
