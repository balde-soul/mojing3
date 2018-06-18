#coding = utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import random

infoq = pd.read_csv('./data/question.csv', index_col=0)
infoc = pd.read_csv('./data/char_embed.txt', header=None, sep=' ', index_col=0)
# char sentence max len
char_fixed_length =


class Data:
    def __init__(self, sources, **options):
        self.train = options.pop('train', None)
        self.test = options.pop('test', None)
        self.val_rate = options.pop('val_rate', None)
        self.batch_size = options.pop('batch_size', None)
        self.char_fixed_length = char_fixed_length
        data = np.array(pd.read_csv(pd.read_csv(sources)))
        if self.train == True:
            assert self.test is None
            assert self.batch_size is not None
            self.x_tr, self.x_te, self.y_tr, self.y_te = \
                train_test_split(data[:, 0], data[:, 1: 3], self.val_rate)
            self.train_batch_num = np.ceil(len(self.y_tr) / self.batch_size)
            self.val_batch_num = np.ceil(len(self.y_te) / self.batch_size)
        if self.test == True:
            assert self.train is None
            assert self.val_rate is None
            assert self.batch_size is None
            self.x, self.y = \
                (data[:, 0], data[:, 1: 3])
        pass

    def gen_data(self, **options):
        """
        产生训练、验证与测试的数据
        训练过程：   训练: training；验证: valing
        测试：
        :param options:
        :return:
        """
        training = options.pop('training', None)
        valing = options.pop('testing', None)
        if self.train is True:
            field = []
            if training:
                assert valing is None
                while 1:
                    if field == []:
                        field = list(range(0, len(self.y_tr)))
                        pass
                    target = random.sample(field)
                    field.remove(target)
                    yield [self.x_tr[target][0], self.x_tr[target][1], self.y_tr[target]]
                    pass
                pass
            if valing:
                assert training is None
                while 1:
                    if field == []:
                        field = list(range(0, len(self.y_te)))
                        pass
                    target = random.sample(field)
                    field.remove(target)
                    yield [self.x_te[target][0], self.x_te[target][1], self.y_te[target]]
                    pass
                pass
            if self.test is True:
                assert training is None
                for i in range(0, len(self.y)):
                    yield np.array([self.x[i], self.y[i]])
                pass
            pass
        if self.test:
            pass
        pass

    def char_symbol(self, question):
        char_symbol = infoq.ix[question].chars.split(' ')
        return char_symbol
        pass

    def embeding_char(self, char_symbol):
        data = np.zeros([self.char_fixed_length, 300], dtype=np.float32)
        row = 0
        for i in char_symbol:
            data[row, :] = infoc.ix[i]
        return data
        pass

    def gen_train(self, **options):
        gen_one = self.gen_data(training=True)
        data1 = np.zeros([self.batch_size, self.char_fixed_length, 300])
        data2 = np.zeros([self.batch_size, self.char_fixed_length, 300])
        label = np.zeros([self.batch_size])
        for i in range(0, self.train_batch_num):
            for j in range(0, self.batch_size):
                q1, q2, y= gen_one.__next__()
                data1[i, :, :] = self.embeding_char(self.char_symbol(q1))
                data2[i, :, :] = self.embeding_char(self.char_symbol(q2))
                label[i] = y
                pass
            pass
        return data1, data2, label
    def gen_val(self):
        gen_one = self.gen_data(valing=True)
        data1 = np.zeros([self.batch_size, self.char_fixed_length, 300])
        data2 = np.zeros([self.batch_size, self.char_fixed_length, 300])
        label = np.zeros([self.batch_size])
        for i in range(0, self.train_batch_num):
            for j in range(0, self.batch_size):
                q1, q2, y = gen_one.__next__()
                data1[i, :, :] = self.embeding_char(self.char_symbol(q1))
                data2[i, :, :] = self.embeding_char(self.char_symbol(q2))
                label[i] = y
                pass
            pass
        return data1, data2, label
        pass

    def gen_test(self):
        pass
