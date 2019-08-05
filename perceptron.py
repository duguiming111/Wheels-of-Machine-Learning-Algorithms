# Author:duguiming
# Description:感知机实现
# Date:2019-08-04
import numpy as np
from data_helper import DataGenerate


class Perceptron(object):
    """采用原始解法"""
    def __init__(self, eta=0.1, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
        self.w = None
        self.b = 0
        self.error_count_history = []

    def fit(self, X_train, y_train):
        """训练"""
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.w, self.b = np.zeros(X_train.shape[1]), 0
        for _ in range(self.n_iter):
            error_count = 0
            # 随机梯度下降法
            # TODO:后续尝试批量梯度下降法和小批量梯度下降法
            for xi, yi in zip(X_train, y_train):
                if yi * self.predict(xi) <= 0:
                    self.w += self.eta * yi * xi
                    self.b += self.eta * yi
                    error_count += 1
            self.error_count_history.append(error_count)
            if error_count == 0:
                break

    def predict_raw(self, x):
        """计算结果向量"""
        return np.dot(x, self.w) + self.b

    def predict(self, x):
        """预测"""
        return np.sign(self.predict_raw(x))

    def evalue(self, X_test, y_test):
        """验证"""
        TP, FN, FP, TN = 0, 0, 0, 0
        for xi, yi in zip(X_test, y_test):
            y_ = self.predict(xi)
            if y_ == 1 and yi == 1:
                TP += 1
            if y_ == 1 and yi == -1:
                FP += 1
            if y_ == -1 and yi == 1:
                FN += 1
            if y_ == -1 and yi == -1:
                TN += 1
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * TP / (2 * TP + FP + FN)
        return accuracy, precision, recall, f1


class PerceptronDual(Perceptron):
    """对偶"""
    def __init__(self, eta=0.1, n_iter=100):
        super(PerceptronDual, self).__init__(eta=eta, n_iter=n_iter)
        self.alpha, self.Gram_matrix = None, None

    def fit(self, X_train, y_train):
        """训练模型"""
        X_train = np.array(X_train)
        n_samples, dim = X_train.shape
        self.alpha, self.w, self.b = np.zeros(n_samples), np.zeros(dim), 0
        # Gram matrix
        self.Gram_matrix = np.dot(X_train, X_train.T)
        # Iteration
        i = 0
        while i < n_samples:
            wx = np.sum(np.dot(self.Gram_matrix[i, :], self.alpha * y_train))
            if y_train[i] * (wx + self.b) <= 0:
                self.alpha[i] += self.eta
                self.b += self.eta * y_train[i]
                i = 0
            else:
                i += 1
        self.w = np.sum(X_train * np.tile((self.alpha * y_train).reshape((n_samples, 1)), (1, dim)), axis=0)


if __name__ == "__main__":
    # step1. 读取数据
    dg = DataGenerate('./data/train.txt')
    X_train, X_test, y_train, y_test = dg.generate()

    # step2. 训练模型
    # model = PerceptronDual(eta=0.1, n_iter=10)
    model = Perceptron(eta=0.1, n_iter=10)
    model.fit(X_train, y_train)

    # step3.测试模型
    accuracy, precision, recall, f1 = model.evalue(X_test, y_test)
    print("准确率: %.2f" % accuracy)
    print("精确率: %.2f" % precision)
    print("召回率: %.2f" % recall)
    print("f1值: %.2f" % f1)
