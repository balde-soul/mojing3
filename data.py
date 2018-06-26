#coding = utf-8
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import random
import sys

#project/version  Data
#./mojing3/v1/* ./Data/mojing3/*
data_root = '../../Data/mojing3/'
infoq = pd.read_csv(data_root + 'question.csv', index_col=0)
infoc = pd.read_csv(data_root + 'char_embed.txt', header=None, sep=' ', index_col=0)
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
        self.sample_num = data.shape[0]
        if self.train is True:
            assert self.test is False, 'test should not be set'
            assert self.batch_size is not None, 'batch size should be specified'
            self.x_tr, self.x_te, self.y_tr, self.y_te = \
                ms.train_test_split(data[:, 1: 3], data[:, 0], test_size=self.val_rate)
            self.train_batch_num = int(np.ceil(len(self.y_tr) / self.batch_size))
            self.val_batch_num = int(np.ceil(len(self.y_te) / self.batch_size))
        if self.test is True:
            assert self.train is False, 'train should not be set'
            assert self.val_rate is None, 'val rate should not be set'
            assert self.batch_size is not None, 'batch size should be specified'
            self.x = data
            self.test_batch_num = int(np.ceil(len(self.x) / self.batch_size))
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
            assert valing != training, 'in train , valing and traing can not be true in the same time'
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
            row += 1
        return data, row - 1
        pass

    def embeding_word(self, ws):
        data = np.zeros([self.word_fixed_length, 300], dtype=np.float32)
        row = 0
        for i in ws:
            data[row, :] = infoc.loc[i]
            row += 1
        return data, row - 1
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
            data1 = np.zeros([self.batch_size, self.char_fixed_length, 300], dtype=np.float32)
            data1_mask = np.zeros([self.batch_size, self.char_fixed_length, 1], dtype=np.float32)
            data2 = np.zeros([self.batch_size, self.char_fixed_length, 300], dtype=np.float32)
            data2_mask = np.zeros([self.batch_size, self.char_fixed_length, 1], dtype=np.float32)
            label = np.zeros([self.batch_size], dtype=np.float32)
            for i in range(0, self.train_batch_num):
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[j, :, :], _mask1 = self.embeding_char(char_symbol(q1))
                    data1_mask[j, _mask1] = 1.0
                    data2[j, :, :], _mask2 = self.embeding_char(char_symbol(q2))
                    data2_mask[j, _mask2] = 1.0
                    label[j] = y
                    pass
                yield data1, data2, label, data1_mask, data2_mask
                pass

        elif word is True:
            data1 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
            data1_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
            data2 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
            data2_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
            label = np.zeros([self.batch_size], dtype=np.float32)
            for i in range(0, self.train_batch_num):
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[j, :, :], _mask1 = self.embeding_word(word_symbol(q1))
                    data1_mask[j, _mask1] = 1.0
                    data2[j, :, :], _mask2 = self.embeding_word(word_symbol(q2))
                    data2_mask[j, _mask2] = 1.0
                    label[j] = y
                    pass
                yield data1, data2, label, data1_mask, data2_mask
                pass
        else:
            print("char or word should be true, gen_train")
            sys.exit()

    def gen_val(self, **options):
        char = options.pop('char', False)
        word = options.pop('word', False)
        assert char & word is not True, "char word is true in the same time"
        gen_one = self.gen_data(valing=True)
        if char is True:
            for i in range(0, self.val_batch_num):
                data1 = np.zeros([self.batch_size, self.char_fixed_length, 300], dtype=np.float32)
                data1_mask = np.zeros([self.batch_size, self.char_fixed_length, 1], dtype=np.float32)
                data2 = np.zeros([self.batch_size, self.char_fixed_length, 300], dtype=np.float32)
                data2_mask = np.zeros([self.batch_size, self.char_fixed_length, 1], dtype=np.float32)
                label = np.zeros([self.batch_size])
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[j, :, :], _mask1 = self.embeding_char(char_symbol(q1))
                    data1_mask[j, _mask1] = 1.0
                    data2[j, :, :], _mask2 = self.embeding_char(char_symbol(q2))
                    data2_mask[j, _mask2] = 1.0
                    label[j] = y
                    pass
                yield data1, data2, label, data1_mask, data2_mask
                pass

        elif word is True:
            for i in range(0, self.val_batch_num):
                data1 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
                data1_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
                data2 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
                data2_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
                label = np.zeros([self.batch_size], dtype=np.float32)
                for j in range(0, self.batch_size):
                    q1, q2, y = gen_one.__next__()
                    data1[j, :, :], _mask1 = self.embeding_word(word_symbol(q1))
                    data1_mask[j, _mask1] = 1.0
                    data2[j, :, :], _mask2 = self.embeding_word(word_symbol(q2))
                    data2_mask[j, _mask2] = 1.0
                    label[j] = y
                    pass
                yield data1, data2, label, data1_mask, data2_mask
                pass

        else:
            print("char or word should be true, gen_val")
        pass

    def gen_test(self, **options):
        assert self.test is True, 'test label has not been set, in __init__, set --test=True'
        char = options.pop('char', False)
        word = options.pop('word', False)
        assert char & word is not True, "char word is true in the same time"
        gen_one = self.gen_data()
        if char:
            for i in range(0, self.sample_num):
                data1 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
                data1_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
                data2 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
                data2_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
                for j in range(0, self.batch_size):
                    q1, q2 = gen_one.__next__()
                    data1[j, :, :], _mask1 = self.embeding_char(char_symbol(q1))
                    data1_mask[j, _mask1] = 1.0
                    data2[j, :, :], _mask2 = self.embeding_char(char_symbol(q2))
                    data2_mask[j, _mask2] = 1.0
                    yield data1, data2, data1_mask, data2_mask
                    pass
                pass
            pass
        if word:
            for i in range(0, self.sample_num):
                data1 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
                data1_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
                data2 = np.zeros([self.batch_size, self.word_fixed_length, 300], dtype=np.float32)
                data2_mask = np.zeros([self.batch_size, self.word_fixed_length, 1], dtype=np.float32)
                for j in range(0, self.batch_size):
                    q1, q2 = gen_one.__next__()
                    data1[j, :, :], _mask1 = self.embeding_word(word_symbol(q1))
                    data1_mask[j, _mask1] = 1.0
                    data2[j, :, :], _mask2 = self.embeding_char(word_symbol(q2))
                    data2_mask[j, _mask2] = 1.0
                    yield data1, data2, data1_mask, data2_mask
                    pass
                pass
            pass
        pass


if __name__ == '__main__':
    data = Data('../../Data/mojing3/train.csv', train=True, batch_size=32, val_rate=0.1)
    gen_train = data.gen_train(char=True)
    gen_val = data.gen_val(char=True)
    x, y, l, mask_x, mask_y = gen_train.__next__()
    for mx in zip(mask_x, x):
        index = np.where(mx[0] == 1.0)
        assert len(index[0]) == 1, 'in element mask train data1 one mask is not 1'
        assert True in (mx[1][index[0] + 1] == 0.0), 'in element train data1, mask index is 0.0'
        assert True in (mx[1][index[0]] != 0.0)
        pass
    for my in zip(mask_y, y):
        index = np.where(my[0] == 1.0)
        assert len(index[0]) == 1, 'in element mask train data1 one mask is not 1'
        assert True in (my[1][index[0] + 1] == 0.0), 'in element train data2, mask index is 0.0'
        assert True in (my[1][index[0]] != 0.0)
        pass

    x, y, l, mask_x, mask_y = gen_val.__next__()
    for mx in zip(mask_x, x):
        index = np.where(mx[0] == 1.0)
        assert len(index[0]) == 1, 'in element mask val data1 one mask is not 1'
        assert True in (mx[1][index[0] + 1] == 0.0), 'in element val data1, mask index is 0.0'
        assert True in (mx[1][index[0]] != 0.0)
        pass
    for my in zip(mask_y, y):
        index = np.where(my[0] == 1.0)
        assert len(index[0]) == 1, 'in element mask train data2 one mask is not 1'
        assert True in (my[1][index[0] + 1] == 0.0), 'in element val data2, mask index is 0.0'
        assert True in (my[1][index[0]] != 0.0)
        pass
