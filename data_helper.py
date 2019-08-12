# Author:duguiming
# Description:数据预处理, BOW特征
# Date:2019-08-04
import jieba
import sklearn
import collections
import itertools
import operator
import array
import sklearn.model_selection


class DataGenerate(object):
    def __init__(self, *args):
        self.data_path = args[0]

    def _fetch_train_test(self, data_path, test_size=0.2):
        """读取数据，划分为训练集和测试集"""
        y = list()
        text_list = list()
        for line in open(data_path, "r", encoding='utf-8').readlines():
            label, text = line[:-1].split('\t', 1)
            text_list.append(list(jieba.cut(text)))
            # 感知机，这里把0标签换成-1标签
            # if int(label) == 0:
            #     label = -1
            y.append(int(label))
        return sklearn.model_selection.train_test_split(text_list, y, test_size=test_size, random_state=1028)

    def _build_dict(self, text_list, min_freq=5):
        """根据输入的文本列表，创建一个最小频词为min_freq的字典，并返回word->wordid"""
        freq_dict = collections.Counter(itertools.chain(*text_list))
        freq_list = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
        words, _ = zip(*filter(lambda wc: wc[1] >= min_freq, freq_list))
        return dict(zip(words, range(len(words))))

    def text2vec(self, text_list, word2id):
        """将输入文本向量化"""
        X = list()
        for text in text_list:
            vect = array.array('l', [0] * len(word2id))
            for word in text:
                if word not in word2id:
                    continue
                vect[word2id[word]] = 1
            X.append(vect)
        return X

    def generate(self):
        """生成数据"""
        # step1.将原始数据拆分成训练集和测试集
        X_train, X_test, y_train, y_test = self._fetch_train_test(self.data_path)

        # step2.创建字典
        word2id = self._build_dict(X_train, min_freq=1)

        # 抽取特征
        X_train = self.text2vec(X_train, word2id)
        X_test = self.text2vec(X_test, word2id)

        return X_train, X_test, y_train, y_test
