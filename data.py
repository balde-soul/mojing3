#coding = utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import random
import sys

infoq = pd.read_csv('./data/question.csv', index_col=0)
infoc = pd.read_csv('./data/char_embed.txt', header=None, sep=' ', index_col=0)
# char sentence max len
char_fixed_length = max([len(i.split(' ')) for i in infoq.chars])
word_fixed_length = max(len(i.split(' ')) for i in infoq.words)


def char_symbol(question):
    cs = infoq.loc[question].chars.split(' ')
    return cs
    pass


def word_symbol(question):
    ws = infoq.loc[question].words.split(' ')
    return ws
    pass


class Data:
    def __init__(self, sources, **options):
        """
        分test与train
        :param sources: 
        :param options: 
        """
        self.train = options.pop('train', None)
        self.test = options.pop('test', None)
        self.val_rate = options.pop('val_rate', None)
        self.batch_size = options.pop('batch_size', None)
        self.char_fixed_length = char_fixed_length
        self.word_fixed_length = word_fixed_length
        data = np.array(pd.read_csv(pd.read_csv(sources)))
        if self.train is True:
            assert self.test is None
            assert self.batch_size is not None
            self.x_tr, self.x_te, self.y_tr, self.y_te = \
                train_test_split(data[:, 0], data[:, 1: 3], self.val_rate)
            self.train_batch_num = np.ceil(len(self.y_tr) / self.batch_size)
            self.val_batch_num = np.ceil(len(self.y_te) / self.batch_size)
        if self.test is True:
            assert self.train is None
            assert self.val_rate is None
            assert self.batch_size is None
            self.x, self.y = \
                (data[:, 0], data[:, 1: 3])
        pass

    def set_batch_size(self, batch_szie):
        self.batch_size = batch_szie

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
                    if field is []:
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
                    if field is []:
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

    def embeding_char(self, cs):
        data = np.zeros([self.char_fixed_length, 300], dtype=np.float32)
        row = 0
        for i in cs:
            data[row, :] = infoc.loc[i]
        return data
        pass

    def embeding_word(self, ws):
        data = np.zeros([self.word_fixed_length, 300], dtype=np.float32)
        row = 0
        for i in ws:
            data[row, :] = infoc.loc[i]
        return data
        pass

    def gen_train(self, **options):
        char = options.pop('char', None)
        word = options.pop('word', None)
        assert char & word is not True, "char word is true in the same time"
        gen_one = self.gen_data(training=True)
        if char is True:
            data1 = np.zeros([self.batch_size, self.char_fixed_length, 300])
            data2 = np.zeros([self.batch_size, self.char_fixed_length, 300])
            label = np.zeros([self.batch_size])
            for i in range(0, self.train_batch_num):
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[i, :, :] = self.embeding_char(char_symbol(q1))
                    data2[i, :, :] = self.embeding_char(char_symbol(q2))
                    label[i] = y
                    pass
                yield data1, data2, label
                pass

        elif word is True:
            data1 = np.zeros([self.batch_size, self.word_fixed_length, 300])
            data2 = np.zeros([self.batch_size, self.word_fixed_length, 300])
            label = np.zeros([self.batch_size])
            for i in range(0, self.train_batch_num):
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[i, :, :] = self.embeding_word(word_symbol(q1))
                    data2[i, :, :] = self.embeding_word(word_symbol(q2))
                    label[i] = y
                    pass
                yield data1, data2, label
                pass
        else:
            print("char or word should be true, gen_train")
            sys.exit()

    def gen_val(self, **options):
        char = options.pop('char', None)
        word = options.pop('word', None)
        assert char & word is not True, "char word is true in the same time"
        gen_one = self.gen_data(valing=True)
        if char is True:
            data1 = np.zeros([self.batch_size, self.char_fixed_length, 300])
            data2 = np.zeros([self.batch_size, self.char_fixed_length, 300])
            label = np.zeros([self.batch_size])
            for i in range(0, self.val_batch_num):
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[i, :, :] = self.embeding_char(char_symbol(q1))
                    data2[i, :, :] = self.embeding_char(char_symbol(q2))
                    label[i] = y
                    pass
                yield data1, data2, label
                pass

        elif word is True:
            data1 = np.zeros([self.batch_size, self.word_fixed_length, 300])
            data2 = np.zeros([self.batch_size, self.word_fixed_length, 300])
            label = np.zeros([self.batch_size])
            for i in range(0, self.val_batch_num):
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[i, :, :] = self.embeding_word(word_symbol(q1))
                    data2[i, :, :] = self.embeding_word(word_symbol(q2))
                    label[i] = y
                    pass
                yield data1, data2, label
                pass

        else:
            print("char or word should be true, gen_val")
        pass

    def gen_test(self):
        pass
