# Author:duguiming
# Description:朴素贝叶斯
# Date:2019-08-06
from collections import defaultdict
import numpy as np
from data_helper import DataGenerate


class NaiveBayesClassificationModel(object):
    def __init__(self):
        # p(类别)
        self.label_prior_prob = dict()
        # P(关键字|类别)
        self.kw_posterior_prob = dict()

    def fit(self, dataset, classes):
        """训练"""
        sub_datasets = defaultdict(lambda: [])
        cls_cnt = defaultdict(lambda: 0)

        for doc_vect, cls in zip(dataset, classes):
            sub_datasets[cls].append(doc_vect)
            cls_cnt[cls] += 1
        # p(类别)
        self.label_prior_prob = {k: v / len(classes) for k, v in cls_cnt.items()}

        # p(关键字|类别)
        dataset = np.array(dataset)
        for cls, sub_dataset in sub_datasets.items():
            sub_dataset = np.array(sub_dataset)
            kw_posterior_prob_vect = np.log((np.sum(sub_dataset, axis=0) + 1) / (np.sum(dataset) + 2))
            self.kw_posterior_prob[cls] = kw_posterior_prob_vect

    def predict(self, doc_vect):
        """单句预测"""
        pred_probs = {}
        for cls, label_prior_prob in self.label_prior_prob.items():
            kw_posterior_prob_vect = self.kw_posterior_prob[cls]
            pred_probs[cls] = np.sum(kw_posterior_prob_vect * doc_vect) + np.log(label_prior_prob)
        return max(pred_probs, key=pred_probs.get)

    def evalue(self, X_test, y_test):
        """验证"""
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
    nb = NaiveBayesClassificationModel()
    nb.fit(X_train, y_train)

    # step3. 测试模型
    accuracy, precision, recall, f1 = nb.evalue(X_test, y_test)
    print("准确率: %.2f" % accuracy)
    print("精确率: %.2f" % precision)
    print("召回率: %.2f" % recall)
    print("f1值: %.2f" % f1)