# coding = utf-8

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import pandas as pd

input_len = 300
output_len = 300
batch = 2
epoch = 100
layer_num = 1
hidden_unit = [100]


def BILSTM(hidden_num, use_dropout):

    def rnn():
        pass
    pass

class Model:
    def __init__(self):
        self.Hidden = [[]]
        self.State = [[]]
        self.batch_size = None
        pass

    def build(self, **options):
        embeding_len = options.pop('embeding_len', None)
        batch_size = options.pop('batch_size', None)
        self.batch_size = batch_size
        num_unit = options.pop('num_unit', [])
        max_tim_step = options.pop('max_time_step', None)
        data_in1 = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_tim_step, embeding_len], name='input-1')
        data_in2 = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_tim_step, embeding_len], name='input-2')
        label = tf.placeholder(dtype=tf.int64, shape=[batch_size], name='label')
        lstm_cell_list1 = []
        lstm_cell_list2 = []
        for unit in num_unit:
            lstm_cell = rnn.BasicLSTMCell(unit, forget_bias=0)
            lstm_cell.zero_state(batch_size, dtype=tf.float32)
            lstm_cell_list1.append(lstm_cell)
        multi = rnn.MultiRNNCell(lstm_cell_list2)
        output1, state1 = tf.nn.dynamic_rnn(multi, data_in1, sequence_length=max_tim_step)
        output2, state2 = tf.nn.dynamic_rnn(multi, data_in2, sequence_length=max_tim_step)
        self.output1 = output1
        self.output2 = output2
        self.label = label


        pass

    def loss(self):
        self.loss = self.label * tf.losses.mean_squared_error(self.output1, self.output2) - \
                    (1 - self.label) * tf.losses.mean_squared_error(self.output1, self.output2)
        self.match_score = tf.losses.mean_squared_error(self.output1, self.output2) / hidden_unit[-1]
        pass

    def train(self, **options):
        epoch = options.pop('epoch', None)
        data = options.pop('data', None)
        save_name = options.pop('save_name', None)
        assert epoch is not None, "epoch"
        assert data is not None, 'data'
        assert save_name is not None, 'save_name'
        step = tf.Variable(0, trainable=False)
        step_up = tf.assign_add(step, 1)
        ema = tf.train.ExponentialMovingAverage(0.99, step)
        moving_op = ema.apply(tf.trainable_variables())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100,filename=save_name)
        tf.summary.scalar(name='loss', tensor=self.loss, collections=tf.global_variables())
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for ep in range(0, epoch):
            for batch in range(data.data.batch_num):
                input1, input2, label = data.gen_train()

        pass

    def eval(self):
        pass
