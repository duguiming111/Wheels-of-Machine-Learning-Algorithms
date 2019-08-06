# Author:duguiming
# Description:KNN
# Date:2019-08-05
import numpy as np
from math import sqrt
from collections import Counter
from data_helper import DataGenerate
import utils.kd_tree as kdtree


class KNNClassifier(object):
    """直接计算距离(线性扫描)"""
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def _predict(self, x):
        """计算K临近数据"""
        distances = [sqrt(np.sum((np.array(x_train) - np.array(x))**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def predict(self, x):
        """预测"""
        y_predict = self._predict(x)
        return y_predict

    def evalue(self, X_test, y_test):
        """模型测试"""
        TP, FP, FN, TN = 0, 0, 0, 0
        for xi, yi in zip(X_test, y_test):
            y_ = self.predict(xi)
            print(y_)
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


class KNNClassifier_KDTree(KNNClassifier):
    """基于kd树"""
    def __init__(self, k):
        super(KNNClassifier_KDTree, self).__init__(k)

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self._X_train = X_train
        self._y_train = y_train
        self.kd_node = kdtree.KDTree(X_train, range(X_train.shape[0]))
        return self

    def predict(self, x):
        """预测"""
        nn_nodes = self.kd_node.search(x, self.k)
        res = [self._y_train[n] for n in nn_nodes]
        label = np.argmax(np.bincount(res))
        return label


if __name__ == "__main__":
    # step1. 读取数据
    dg = DataGenerate('./data/train.txt')
    X_train, X_test, y_train, y_test = dg.generate()

    # step2. 训练模型
    # kc = KNNClassifier(k=3)
    kc = KNNClassifier_KDTree(k=3)
    model = kc.fit(X_train, y_train)

    # step3. 测试模型
    accuracy, precision, recall, f1 = model.evalue(X_test, y_test)
    print("准确率: %.2f" % accuracy)
    print("精确率: %.2f" % precision)
    print("召回率: %.2f" % recall)
    print("f1值: %.2f" % f1)