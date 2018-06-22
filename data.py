#coding = utf-8
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import random
import sys

infoq = pd.read_csv('./Data/question.csv', index_col=0)
infoc = pd.read_csv('./Data/char_embed.txt', header=None, sep=' ', index_col=0)
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
        self.train = options.pop('train', False)
        self.test = options.pop('test', False)
        self.val_rate = options.pop('val_rate', None)
        self.batch_size = options.pop('batch_size', None)
        self.char_fixed_length = char_fixed_length
        self.word_fixed_length = word_fixed_length
        data = np.array(pd.read_csv(sources))
        if self.train is True:
            assert self.test is False
            assert self.batch_size is not None
            self.x_tr, self.x_te, self.y_tr, self.y_te = \
                ms.train_test_split(data[:, 1: 3], data[:, 0], test_size=self.val_rate)
            self.train_batch_num = int(np.ceil(len(self.y_tr) / self.batch_size))
            self.val_batch_num = int(np.ceil(len(self.y_te) / self.batch_size))
        if self.test is True:
            assert self.train is False
            assert self.val_rate is None
            assert self.batch_size is None
            self.x = data
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
        training = options.pop('training', False)
        valing = options.pop('valing', False)
        if self.train is True:
            assert valing != training is True, 'in train , valing and traing can not be true in the same time'
            field = []
            if training:
                while 1:
                    if len(field) is 0:
                        field = list(range(0, len(self.y_tr)))
                        pass
                    target = random.choice(field)
                    field.remove(target)
                    yield self.x_tr[target][0], self.x_tr[target][1], self.y_tr[target]
                    pass
                pass
            if valing:
                while 1:
                    if len(field) is 0:
                        field = list(range(0, len(self.y_te)))
                        pass
                    target = random.choice(field)
                    field.remove(target)
                    yield self.x_te[target][0], self.x_te[target][1], self.y_te[target]
                    pass
                pass
        if self.test:
            field = []
            assert valing is False, 'in test ,valing is illegal'
            assert training is False, 'in test ,valing is illegal'
            i = 0
            while 1:
                temp_i = i
                i += 1
                if i == len(self.x):
                    i = 0
                else:
                    pass
                yield self.x[temp_i][0], self.x[temp_i][1]
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
        """
        
        :param options: 
        :keyword char :若使用char
        :keyword word:若使用word
        :return: 
        """
        char = options.pop('char', False)
        word = options.pop('word', False)
        assert (char != word) is True, "char word is true in the same time"
        gen_one = self.gen_data(training=True)
        if char is True:
            data1 = np.zeros([self.batch_size, self.char_fixed_length, 300])
            data2 = np.zeros([self.batch_size, self.char_fixed_length, 300])
            label = np.zeros([self.batch_size])
            for i in range(0, self.train_batch_num):
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[j, :, :] = self.embeding_char(char_symbol(q1))
                    data2[j, :, :] = self.embeding_char(char_symbol(q2))
                    label[j] = y
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
                    data1[j, :, :] = self.embeding_word(word_symbol(q1))
                    data2[j, :, :] = self.embeding_word(word_symbol(q2))
                    label[j] = y
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
                    data1[j, :, :] = self.embeding_char(char_symbol(q1))
                    data2[j, :, :] = self.embeding_char(char_symbol(q2))
                    label[j] = y
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
                    data1[j, :, :] = self.embeding_word(word_symbol(q1))
                    data2[j, :, :] = self.embeding_word(word_symbol(q2))
                    label[j] = y
                    pass
                yield data1, data2, label
                pass

        else:
            print("char or word should be true, gen_val")
        pass

    def gen_test(self):
        pass


if __name__ == '__main__':
    data = Data('./Data/train.csv', train=True, batch_size=32, val_rate=0.1)
    gen_train = data.gen_train(char=True)
    gen_val = data.gen_val(char=True)
    # b = np.zeros(shape=[3, 4, 5])
    batch_num = 0
    while True:
        try:
            x, y, l = gen_train.__next__()
            batch_num += 1
            print(batch_num)
        except Exception:
            print(Exception.__name__)
            break
            pass
        pass
    pass
