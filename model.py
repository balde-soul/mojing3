# coding = utf-8

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import time
import Putil.tf.ops_process as tfop

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
        self.standard_loss = None

        self.device = True
        self.basic_lr = 0.0001
        self.retrain = False
        self.retrain_file = ''
        self.save_path = ''
        self.epoch = None
        self.val_while_n_epoch = 1
        self.save_while_n_step = 10
        self.display_while_n_step = 10

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
        self.input1 = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_time_step, self.embeding_len],
                                     name='input-1')
        self.input2 = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_time_step, self.embeding_len],
                                     name='input-2')
        lstm_cell_list = []
        for unit in self.hidden_unit:
            lstm_cell = rnn.BasicLSTMCell(unit, forget_bias=0, activation=tf.tanh)
            lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            lstm_cell_list.append(lstm_cell)
        multi = rnn.MultiRNNCell(lstm_cell_list)
        # output [batch_size, max_time_step, hidden_unit[-1]]
        output1, self.state1 = tf.nn.dynamic_rnn(multi, self.input1, dtype=tf.float32)
        output2, self.state2 = tf.nn.dynamic_rnn(multi, self.input2, dtype=tf.float32)
        # self.output1 = tf.nn.softmax(output1, axis=-1)
        # 对使用tanh激活的输出output1进行归一化，由于使用的是余弦距离衡量相似度，我们将向量的模长归一，
        # output1代表的是包含所有step的最后层输出[batch, time_step, output_len]
        #
        self.output1 = tf.nn.l2_normalize(output1, axis=2)
        self.output2 = tf.nn.l2_normalize(output2, axis=2)

        self.output1_mask = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_time_step, 1])
        self.output2_mask = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_time_step, 1])

        self.apply_mask1 = tf.reduce_sum(self.output1 * self.output1_mask, axis=1)
        self.apply_mask2 = tf.reduce_sum(self.output2 * self.output1_mask, axis=1)

        self.cosine_coss = tf.div(tf.reduce_sum(tf.multiply(self.apply_mask1, self.apply_mask2), axis=-1),
                                  tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.apply_mask1), axis=-1)),
                                              tf.sqrt(tf.reduce_sum(tf.square(self.apply_mask2), axis=-1))))

        self.match_score = tf.multiply(0.5, (1 + tf.reduce_mean(self.cosine_coss)), name='match_score')
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
        self.label = tf.placeholder(dtype=tf.float32, shape=[self.batch_size], name='label')

        self.loss = tf.multiply(0.5, (1 + tf.reduce_mean(self.cosine_coss - 2 * self.label * self.cosine_coss)),
                                name='cosine_loss')
        _match_score = 0.5 * (1 + self.cosine_coss)
        self.standard_loss = tf.reduce_mean(tf.losses.log_loss(self.label, _match_score, epsilon=1e-15),
                                            name='log_loss')
        tf.add_to_collection(
            self.TRAIN_C,
            tf.summary.scalar(name='tr_loss', tensor=self.loss))
        tf.add_to_collection(
            self.VAL_C,
            tf.summary.scalar(name='v_loss', tensor=self.loss))
        tf.add_to_collection(
            self.VAL_C,
            tf.summary.scalar(name='v_match_loss', tensor=self.standard_loss))
        tf.add_to_collection(
            self.TRAIN_C,
            tf.summary.scalar(name='tr_match_loss', tensor=self.standard_loss))
        pass

    def deploy(self):
        self.test_match_score = 0.5 * (1 + self.cosine_coss)
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
        :keyword basic_lr: special the basic learning rate, default is 0.0001
        :return: 
        """
        self.device = options.pop('device', self.device)
        if self.device is False:
            import os
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            pass
        else:
            pass
        self.retrain = options.pop('retrain', self.retrain)
        self.retrain_file = options.pop('retrain_file', self.retrain_file)
        if self.retrain is True:
            assert self.retrain_file is not None, 'retrain_file, train'
        self.save_path = options.pop('save_path', self.save_path)
        assert self.save_path != '', 'save_path, train'
        self.epoch = options.pop('epoch', self.epoch)
        self.basic_lr = options.pop('epoch', self.basic_lr)
        self.val_while_n_epoch = options.pop('val_while_n_epoch', self.val_while_n_epoch)
        self.save_while_n_step = options.pop('save_while_n_step', self.save_while_n_step)
        self.display_while_n_step = options.pop('display_shilw_n_step', self.display_while_n_step)
        code_test = options.pop('code_test', False)
        data = options.pop('data', None)
        char = options.pop('char', False)
        word = options.pop('word', False)
        assert self.epoch is not None, "epoch"
        assert data is not None, 'data'

        # record step
        step = tf.Variable(0, trainable=False)
        # step_up = tf.assign_add(step, 1)

        # opt
        opt = tf.train.AdamOptimizer(self.basic_lr)
        train_target = self.standard_loss
        print('training target loss: ' + train_target.name)
        mini = opt.minimize(train_target, global_step=step)

        # moving average
        ema = tf.train.ExponentialMovingAverage(0.99, step)
        with tf.control_dependencies([mini]):
            train_op = ema.apply(tf.trainable_variables())
            pass

        train_summary = tf.summary.merge_all(self.TRAIN_C)
        val_summary = tf.summary.merge_all(self.VAL_C)

        # saveer
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100, filename=save_name)

        # if self.device is False:
        #     sess = tf.Session(config = tf.ConfigProto(device_count={'/cpu': 0}))
        # else:
        #     sess = tf.Session()
        #     pass
        sess = tf.Session()
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:700")

        writer = tf.summary.FileWriter(self.save_path + '-summary-', graph=sess.graph)

        sess.run(tf.global_variables_initializer())

        # continue train
        if self.retrain:
            saver.restore(sess, self.retrain_file)

        Epoch = 0

        for ep in range(0, self.epoch):
            print('>>------------Epoch------------<<')
            gen = data.gen_train(char=char, word=word)
            epoch_standard_loss=[]
            epoch_loss=[]
            while True:
                try:
                    start_data_read = time.time()
                    input1, input2, label, mask1, mask2 = gen.__next__()
                    end_data_read = time.time()
                except Exception:
                    break
                start_train = time.time()
                _, loss, match_score, standard_loss, summary = \
                    sess.run(
                        [train_op, self.loss, self.match_score, self.standard_loss, train_summary],
                        feed_dict={
                            self.input1: input1,
                            self.input2: input2,
                            self.label: label,
                            self.output1_mask: mask1,
                            self.output2_mask: mask2
                        }
                    )
                end_train = time.time()
                epoch_loss.append(loss)
                epoch_standard_loss.append(standard_loss)
                # for test the code correct or not
                if code_test:
                    print('>>----------val-----------<<:')
                    genv = data.gen_val(char=char, word=word)
                    for i in range(0, 10):
                        try:
                            start_data_read = time.time()
                            input1, input2, label, mask1, mask2 = genv.__next__()
                            end_data_read = time.time()
                        except Exception:
                            break
                        start_train = time.time()
                        loss, match_score, standard_loss, summary = \
                            sess.run(
                                [self.loss, self.match_score, self.standard_loss, val_summary],
                                feed_dict={
                                    self.input1: input1,
                                    self.input2: input2,
                                    self.label: label,
                                    self.output1_mask: mask1,
                                    self.output2_mask: mask2
                                }
                            )
                        end_train = time.time()
                        sess.run(step)
                        print("code_test<<: step: {0}"
                              ", epoch: {1}, loss: {2}"
                              ", match_score: {3}"
                              ", now_mean_standard_loss: {6}"
                              ", now_mean_loss"
                              ", train_time: {4}s"
                              ", data_read_batch_time: {5}s"
                              .format(sess.run(step), Epoch, loss, match_score, end_train - start_train,
                                      end_data_read - start_data_read, standard_loss))
                        pass
                    pass

                # training step display
                if sess.run(step) % self.display_while_n_step == 0:
                    print(
                        "train_step<<: step: {0}"
                        ", epoch: {1}, loss: {2}"
                        ", match_score: {3}"
                        ", now_mean_standard_loss: {6}"
                        ", now_mean_loss: {7}"
                        ", train_time: {4}s"
                        ", data_read_batch_time: {5}s"
                            .format(sess.run(step), Epoch, loss, match_score, end_train - start_train,
                                    end_data_read - start_data_read, np.mean(epoch_standard_loss),
                                    np.mean(epoch_loss)))
                    pass

                if sess.run(step) % self.save_while_n_step == 0:
                    writer.add_summary(summary, global_step=sess.run(step))
                    saver.save(sess, self.save_path + str(sess.run(step)), global_step=sess.run(step), write_meta_graph=True)
                    pass
                pass

            # one train epoch display
            print('train_epoch<<: epoch: {0}'
                  ', epoch_loss: {1}'
                  ', epoch_standard_loss: {2}'
                  .format(ep, np.mean(epoch_loss),
                          np.mean(epoch_standard_loss)))

            if ep % self.val_while_n_epoch == 0:
                print('>>----------val-----------<<:')
                gen = data.gen_val(char=char, word=word)
                epoch_loss = []
                epoch_standard_loss = []
                v_step = 0
                summary = None
                while True:
                    try:
                        start_data_read = time.time()
                        input1, input2, label, mask1, mask2 = gen.__next__()
                        end_data_read = time.time()
                    except Exception:
                        break
                    start_train = time.time()
                    loss, match_score, standard_loss, summary = \
                        sess.run(
                            [self.loss, self.match_score, self.standard_loss, val_summary],
                            feed_dict={
                                self.input1: input1,
                                self.input2: input2,
                                self.label: label,
                                self.output1_mask: mask1,
                                self.output2_mask: mask2
                            }
                        )
                    epoch_loss.append(loss)
                    epoch_standard_loss.append(standard_loss)
                    end_train = time.time()
                    v_step += 1

                    # val step display
                    if v_step % self.display_while_n_step == 0:
                        print(
                            "val_step<<: step: {0}"
                            ", epoch: {1}"
                            ", loss: {2}"
                            ", match_score: {3}"
                            ", now_mean_standard_loss: {6}"
                            ", now_mean_loss: {7}"
                            ", train_time: {4}s"
                            ", data_read_batch_time: {5}s"
                                .format(v_step, Epoch, loss, match_score, end_train - start_train,
                                    end_data_read - start_data_read, np.mean(epoch_standard_loss), np.mean(epoch_loss)))
                        pass
                    pass
                # one val epoch display
                print(
                    'val_epoch<<: epoch: {0}'
                    ', epoch_standard_loss: {1}'
                    ', epoch_loss: {2}'
                        .format(ep, np.mean(epoch_standard_loss),
                                np.mean(np.mean(epoch_loss))))
                writer.add_summary(summary, global_step=sess.run(step))
                saver.save(sess, self.save_path + '-v-', global_step=sess.run(step), write_meta_graph=True)
                pass
            Epoch += 1

            pass
        pass

    def restore(self, **options):
        restore_file = options.pop('weight', None)
        restore_moving = options.pop('moving', False)
        data = options.pop('data', None)


        assert data is not None, 'data not special'
        assert data.test is True, 'data is not test data'
        sess = tf.Session()
        if restore_moving:
            saver = tf.train.Saver(tfop.original_apply_moving(sess))
        else:
            saver = tf.train.Saver()
            saver.restore(sess, restore_file)
            pass
        return sess
        pass

    def test(self, sess, **options):
        save_path = options.pop('save_path', '')
        assert save_path != '', 'should specify save path'
        data = options.pop('data', None)
        assert data is not None, 'should set data'
        char = options.pop('char', False)
        word = options.pop('word', False)
        assert char == word, 'word and char can not be the same'
        data_gen = data.gen_test(word=word, char=char)
        time_coss = list()
        match_score = list()
        while True:
            try:
                data1, data2, data1_mask, data2_mask = data_gen.__next__()
                pass
            except Exception:
                print('gen data error')
                break
                pass
            begin_time = time.time()
            match_score.append(list(sess.run([self.test_match_score])))
            end_time = time.time()
            time_coss.append(end_time - begin_time)
            pass
        return np.reshape(np.array(match_score), [-1])
        pass
    pass


if __name__ == '__main__':
    import data
    model = Model()
    train_data_handle = data.Data(sources='../../Data/mojing3/train.csv', batch_size=32, val_rate=0.1, train=True)
    model.build(embeding_len=300, batch_size=32, hidden_unit=[200, 50],
                max_time_step=train_data_handle.char_fixed_length)
    model.build_loss()
    model.train(epoch=100, save_path='./check/', save_while_n_step=10000, val_while_n_epoch=2,
                data=train_data_handle, char=True, display_shilw_n_step=1, code_test=True, device=False)
    sess=tf.Session()
    sess.close()
    tf.reset_default_graph()
    test_data_handle = data.Data(sources='../../Data/mojing3/test.csv', batch_size=32, test=True)
    model.build(embeding_len=300, batch_size=32, hidden_unit=[200, 50],
                max_time_step=train_data_handle.char_fixed_length)
