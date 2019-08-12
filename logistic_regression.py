# Author:duguiming
# Description:逻辑回归
# Date:2019-08-06
import numpy as np
from data_helper import DataGenerate


class LogisticRegressionModel(object):
    def __init__(self):
        self.W = None

    def sigmoid(self, x):
        """sigmoid函数"""
        #  sigmoid: 1 / (1 + e ** -x)
        return 1 / (1 + np.exp(-x))

    def loss(self, h, y_train, num_train):
        """损失函数"""
        # cross entropy: 1/m * sum((y*np.log(h) + (1-y)*np.log((1-h))))
        return -np.sum(y_train * np.log(h) + (1 - y_train) * np.log(1 - h)) / num_train

    def update(self, X_train_, h, y_train, num_train, learning_rate):
        """参数更新"""
        #  dW = cross entropy' = 1/m * sum(h-y) * x
        dW = X_train_.T.dot(h - y_train) / num_train
        #  W = W - learning_rate*dW
        self.W -= learning_rate * dW

    def fit(self, X_train, y_train, learning_rate=0.1, num_iters=10000):
        """训练模型"""
        X_train = np.array(X_train)
        num_train, dim_feature = X_train.shape
        y_train = np.array(y_train).reshape((num_train, 1))
        X_train_ = np.hstack((X_train, np.ones((num_train, 1))))
        self.W = 0.001 * np.random.randn(dim_feature + 1, 1)
        loss_history = []
        for i in range(num_iters + 1):
            g = np.dot(X_train_, self.W)
            h = self.sigmoid(g)
            loss = self.loss(h, y_train, num_train)
            loss_history.append(loss)
            self.update(X_train_, h, y_train, num_train, learning_rate)
            if i % 100 == 0:
                print('Iters: %r/%r Loss: %r' % (i, num_iters, loss))

    def predict(self, doc_vec):
        """单句预测"""
        tmp = np.array(doc_vec)
        doc_vec = tmp.reshape(1, tmp.shape[0])
        doc_vec = np.hstack((doc_vec, np.ones((1, 1))))
        g = np.dot(doc_vec, self.W)
        h = self.sigmoid(g)
        if h >= 0.5:
            prob = 1
        else:
            prob = 0
        return prob

    def evalue(self, X_test, y_test):
        """测试模型"""
        TP, FP, FN, TN = 0, 0, 0, 0
        for xi, yi in zip(X_test, y_test):
            y_ = self.predict(xi)
            if y_ == 1 and yi == 1:
                TP += 1
            if y_ == 1 and yi == 0:
                FP += 1
            if y_ == 0 and yi == 1:
                FN += 1
            if y_ == 0 and yi == 0:
                TN += 1
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * TP / (2 * TP + FP + FN)
        return accuracy, precision, recall, f1


if __name__ == "__main__":
    # step1. 读取数据
    dg = DataGenerate('./data/train.txt')
    X_train, X_test, y_train, y_test = dg.generate()

    # step2. 训练模型
    lr = LogisticRegressionModel()
    lr.fit(X_train, y_train, num_iters=100)

    # step3. 测试模型
    accuracy, precision, recall, f1 = lr.evalue(X_test, y_test)
    print("准确率: %.2f" % accuracy)
    print("精确率: %.2f" % precision)
    print("召回率: %.2f" % recall)
    print("f1值: %.2f" % f1)