# coding = utf-8

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import pandas as pd
import time

input_len = 300
output_len = 300
save_name = 'lstm'


def BILSTM(hidden_num, use_dropout):

    def rnn():
        pass
    pass


class Model:
    def __init__(self, **options):
        self.Hidden = [[]]
        self.State = [[]]

        self.embeding_len = None
        self.batch_size = None
        self.max_time_step = None
        self.hidden_unit = list()
        self.label = None
        self.state1 = None
        self.state2 = None
        self.output1 = None
        self.output2 = None
        self.input1 = None
        self.input2 = None
        self.label = None

        self.match_score = None
        self.loss = None

        self.retrain = False
        self.retrain_file = ''
        self.save_path = ''
        self.epoch = None
        self.val_while_n_epoch = 1
        self.save_while_n_step = 10

        self.TRAIN_C = 'TRAIN'
        self.VAL_C = 'VAL'
        self.TEST_C = 'TEST'
        tf.GraphKeys.TRAIN = 'TRAIN'
        tf.GraphKeys.VAL = 'VAL'
        tf.GraphKeys.TEST = 'TEST'
        pass

    def three_summary_add(self, summary_op):
        tf.add_to_collection(
            self.TRAIN_C,
            summary_op
        )
        tf.add_to_collection(
            self.TEST_C,
            summary_op
        )
        tf.add_to_collection(
            self.VAL_C,
            summary_op
        )

    def build(self, **options):
        """
        embeding_len : 词向量的长度
        batch_size : 批量
        hidden_unit : list 每一层的隐层单元数
        max_time_step : 最长的序列长度
        :param options: 
        :return: 
        """
        self.embeding_len = options.pop('embeding_len', self.embeding_len)
        self.batch_size = options.pop('batch_size', self.batch_size)
        self.hidden_unit = options.pop('hidden_unit', self.hidden_unit)
        self.max_time_step = options.pop('max_time_step', self.max_time_step)
        self.batch_size = self.batch_size
        self.input1 = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_time_step, self.embeding_len], name='input-1')
        self.input2 = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_time_step, self.embeding_len], name='input-2')
        self.label = tf.placeholder(dtype=tf.float32, shape=[self.batch_size], name='label')
        lstm_cell_list = []
        for unit in self.hidden_unit:
            lstm_cell = rnn.BasicLSTMCell(unit, forget_bias=0, activation=tf.tanh)
            lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            lstm_cell_list.append(lstm_cell)
        multi = rnn.MultiRNNCell(lstm_cell_list)
        output1, self.state1 = tf.nn.dynamic_rnn(multi, self.input1, dtype=tf.float32)
        output2, self.state2 = tf.nn.dynamic_rnn(multi, self.input2, dtype=tf.float32)
        # self.output1 = tf.nn.softmax(output1, axis=-1)
        # 对使用tanh激活的输出output1进行归一化，由于使用的是余弦距离衡量相似度，我们将向量的模长归一，
        # output1代表的是包含所有step的最后层输出[batch, time_step, output_len]
        #
        self.output1 = tf.nn.l2_normalize(output1, axis=2)
        self.output2 = tf.nn.l2_normalize(output2, axis=2)
        # self.output1 = tf.transpose(
        #     tf.div(tf.transpose(output1, [2, 0, 1]), tf.reduce_mean(tf.square(output1), axis=2)), [1, 2, 0])
        # # self.output2 = tf.nn.softmax(output1, axis=-1)
        # self.output2 = tf.transpose(
        #     tf.div(tf.transpose(output2, [2, 0, 1]), tf.reduce_mean(tf.square(output2), axis=2)), [1, 2, 0])

        self.three_summary_add(
            tf.summary.image(name='output1', tensor=tf.cast(
                255 * tf.div(tf.expand_dims(tf.expand_dims(self.output1[:, -1], axis=[0]), axis=[3]) + 1, 2),
                tf.uint8)))
        self.three_summary_add(
            tf.summary.image(name='output2', tensor=tf.cast(
                255 * tf.div(tf.expand_dims(tf.expand_dims(self.output2[:, -1], axis=[0]), axis=[3]) + 1, 2),
                tf.uint8)))

        # 分离四个gate， 分别summary
        for i in lstm_cell_list:
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[0].name + '-gate-weight-1',
                                     values=i.weights[0][:, 0: i.state_size.c]))
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[1].name + '-gate-bias-1',
                                     values=i.weights[1][0: i.state_size.c]))
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[0].name + '-gate-weight-2',
                                     values=i.weights[0][:, i.state_size.c: 2 * i.state_size.c]))
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[1].name + '-gate-bias-2',
                                     values=i.weights[1][i.state_size.c: 2 * i.state_size.c]))
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[0].name + '-gate-weight-3',
                                     values=i.weights[0][:, 2 * i.state_size.c: 3 * i.state_size.c]))
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[1].name + '-gate-bias-3',
                                     values=i.weights[1][2 * i.state_size.c: 3 * i.state_size.c]))
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[0].name + '-gate-weight-4',
                                     values=i.weights[0][:, 3 * i.state_size.c: 4 * i.state_size.c]))
            tf.add_to_collection(
                self.TRAIN_C,
                tf.summary.histogram(name=i.weights[1].name + '-gate-bias-4',
                                     values=i.weights[1][3 * i.state_size.c: 4 * i.state_size.c]))
        pass

    def build_loss(self):
        cosine_coss = tf.div(tf.reduce_sum(tf.multiply(self.output1[:, -1, :], self.output2[:, -1, :]), axis=-1),
                             tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.output1[:, -1, :]), axis=-1)),
                                         tf.sqrt(tf.reduce_sum(tf.square(self.output2[:, -1, :]), axis=-1))))
        # singal_loss = 0.5 * (self.label * (1 - cosine_coss) + (1 - self.label) * (1 + cosine_coss))
        self.loss = 0.5 * (1 + tf.reduce_mean(cosine_coss - 2 * self.label * cosine_coss))
        self.match_score = 0.5 * (1 + tf.reduce_mean(cosine_coss))
        tf.add_to_collection(
            self.TRAIN_C,
            tf.summary.scalar(name='loss', tensor=self.loss))
        tf.add_to_collection(
            self.TRAIN_C,
            tf.summary.scalar(name='loss', tensor=self.loss))
        tf.add_to_collection(
            self.VAL_C,
            tf.summary.scalar(name='match_loss', tensor=self.match_score))
        tf.add_to_collection(
            self.TRAIN_C,
            tf.summary.scalar(name='match_loss', tensor=self.match_score))
        tf.add_to_collection(
            self.TEST_C,
            tf.summary.scalar(name='match_loss', tensor=self.match_score))
        tf.add_to_collection(
            self.TEST_C,
            tf.summary.scalar(name='match_loss', tensor=self.match_score))
        pass

    def train(self, **options):
        """
        
        :param options: 
        :keyword retrain: when continue train use retrain=True
        :keyword retrain_file: when continue train use retrain_file to special restore weight file
        :keyword save_path: special the path to save summary and checkpoint
        :keyword epoch: special epoch to tain
        :keyword val_while_n_epoch: special val between n epoch
        :keyword save_shile_n_step: special save summary and checkpoint between n epoch
        :keyword data: special Data obj to handle generator data input feed
        :keyword cahr: special use data to generate char embeding, set True
        :keyword word: special use data to generate word embeding, set True
        :return: 
        """
        self.retrain = options.pop('retrain', self.retrain)
        self.retrain_file = options.pop('retrain_file', self.retrain_file)
        if self.retrain is True:
            assert self.retrain_file is not None, 'retrain_file, train'
        self.save_path = options.pop('save_path', self.save_path)
        assert self.save_path != '', 'save_path, train'
        self.epoch = options.pop('epoch', self.epoch)
        self.val_while_n_epoch = options.pop('val_while_n_epoch', self.val_while_n_epoch)
        self.save_while_n_step = options.pop('save_while_n_step', self.save_while_n_step)
        data = options.pop('data', None)
        char = options.pop('char', False)
        word = options.pop('word', False)
        assert self.epoch is not None, "epoch"
        assert data is not None, 'data'

        # record step
        step = tf.Variable(0, trainable=False)
        step_up = tf.assign_add(step, 1)

        # opt
        opt = tf.train.AdamOptimizer(0.01)
        mini = opt.minimize(self.loss, global_step=step)

        # moving average
        ema = tf.train.ExponentialMovingAverage(0.99, step)
        with tf.control_dependencies([mini]):
            train_op = ema.apply(tf.trainable_variables())
            pass

        train_summary = tf.summary.merge_all(self.TRAIN_C)
        val_summary = tf.summary.merge_all(self.VAL_C)
        writer = tf.summary.FileWriter(self.save_path + '-summary-')

        # saveer
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100, filename=save_name)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        Epoch = 0
        for ep in range(0, self.epoch):
            print('>>------------Epoch------------<<')
            gen = data.gen_train(char=char, word=word)
            while True:
                try:
                    start_data_read = time.time()
                    input1, input2, label = gen.__next__()
                    end_data_read = time.time()
                except Exception:
                    break
                start_train = time.time()
                _, loss, match_score, summary = \
                    sess.run(
                        [train_op, self.loss, self.match_score, train_summary],
                        feed_dict={
                            self.input1: input1,
                            self.input2: input2,
                            self.label: label
                        }
                    )
                # _, loss, match_score = \
                #     sess.run(
                #         [train_op, self.loss, self.match_score],
                #         feed_dict={
                #             self.input1: input1,
                #             self.input2: input2,
                #             self.label: label
                #         }
                #     )

                # print('>>----------val-----------<<:')
                # genv = data.gen_val(char=char, word=word)
                # while True:
                #     try:
                #         start_data_read = time.time()
                #         input1, input2, label = genv.__next__()
                #         end_data_read = time.time()
                #     except Exception:
                #         break
                #     start_train = time.time()
                #     _, loss, match_score, summary = \
                #         sess.run(
                #             [train_op, self.loss, self.match_score, val_summary],
                #             feed_dict={
                #                 self.input1: input1,
                #                 self.input2: input2,
                #                 self.label: label
                #             }
                #         )
                #     end_train = time.time()
                #     sess.run(step)
                #     print("<<: step: {0}, epoch: {1}, loss: {2}, match_score: {3}, "
                #           "train_time: {4}s, data_read_batch_time: {5}s"
                #           .format(sess.run(step), Epoch, loss, match_score, end_train - start_train,
                #                   end_data_read - start_data_read))

                end_train = time.time()
                print("<<: step: {0}, epoch: {1}, loss: {2}, match_score: {3}, "
                      "train_time: {4}s, data_read_batch_time: {5}s"
                      .format(sess.run(step), Epoch, loss, match_score, end_train - start_train,
                              end_data_read - start_data_read))
                if (sess.run(step) % self.save_while_n_step == 0) and (sess.run(step) != 0):
                    writer.add_summary(summary, global_step=sess.run(step))
                    saver.save(sess, self.save_path + '-ckpt-', global_step=sess.run(step), write_meta_graph=True)
                pass

            if (ep % self.val_while_n_epoch == 0) and (ep != 0):
                print('>>----------val-----------<<:')
                gen = data.gen_val(char=char, word=word)
                while True:
                    try:
                        start_data_read = time.time()
                        input1, input2, label = gen.__next__()
                        end_data_read = time.time()
                        start_train = time.time()
                        _, loss, match_score, summary = \
                            sess.run(
                                [train_op, self.loss, self.match_score, val_summary],
                                feed_dict={
                                    self.input1: input1,
                                    self.input2: input2,
                                    self.label: label
                                }
                            )
                        end_train = time.time()
                    except Exception:
                        break
                    sess.run(step)
                    print("<<: step: {0}, epoch: {1}, loss: {2}, match_score: {3}, "
                          "train_time: {4}s, data_read_batch_time: {5}s"
                          .format(sess.run(step), Epoch, loss, match_score, end_train - start_train,
                                  end_data_read - start_data_read))

            Epoch += 1
        pass

    def restore(self, **options):
        restore_file = options.pop('weight', None)
        restore_moving = options.pop('moving', False)
        pass

    def test(self):
        test_summary = tf.summary.merge_all(self.TEST_C)
        pass
    pass


if __name__ == '__main__':
    import data
    model = Model()
    train_data_handle = data.Data(sources='./Data/train.csv', batch_size=32, val_rate=0.1, train=True)
    model.build(embeding_len=300, batch_size=32, hidden_unit=[200, 50],
                max_time_step=train_data_handle.char_fixed_length)
    model.build_loss()
    model.train(epoch=100, save_path='./check/', save_while_n_step=10000, val_while_n_epoch=2,
                data=train_data_handle, char=True)
