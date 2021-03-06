# coding = utf-8
import model
import data
from optparse import OptionParser

embeding_len = 300
parser = OptionParser(usage="usage:%prog [options] arg1 arg2")
parser.add_option(
    '--dt',
    '--data_type',
    action='store',
    type=str,
    dest='DataType',
    default='char',
    help='set data type, char or word, default char'

)
parser.add_option(
    '--vr',
    '--val_rate',
    action='store',
    type=float,
    dest='ValRate',
    default=0.1,
    help='train val rate set,default 0.1, val rate should be smaller than 0.5'
)
parser.add_option(
    '--m',
    '--model',
    action='store',
    type=str,
    dest='Mode',
    default='train',
    help='choice train or test, default train'
)
parser.add_option(
    '--bs',
    '--batch_size',
    action='store',
    type=int,
    dest='BatchSize',
    default=32,
    help='set batch size , default 32'
)
parser.add_option(
    '--sp',
    '--save_path',
    action='store',
    type=str,
    dest='SavePath',
    default='./check/',
    help='special the path to save model and summary'
)
parser.add_option(
    '--hu',
    '--hidden_unit',
    action='store',
    type=str,
    dest='HiddenUnit',
    default='200 50',
    help='set the hidden unit ,default 200, 50'
)
parser.add_option(
    '--ep',
    '--epoch',
    action='store',
    type=int,
    dest='Epoch',
    default=200,
    help='set train epoch, default 1000'
)
parser.add_option(
    '--cpu',
    action='store_false',
    dest='UseCpu',
    default=True,
    help='special to use cpu'
)
parser.add_option(
    '--br',
    '--basic_lr',
    action='store',
    type=float,
    dest='LearningRate',
    default=0.0001,
    help='set learning rate, default 0.0001'
)
parser.add_option(
    '--ds',
    '--display_while_n_step',
    action='store',
    type=int,
    dest='DisplayWhileNStep',
    default=32,
    help='while n step, would display info, default 20'
)
parser.add_option(
    '--ss',
    '--save_while_n_step',
    action='store',
    type=int,
    dest='SaveWhileNStep',
    default=1000,
    help='while n step, would save summary and checkpoint, default 10000'
)
parser.add_option(
    '--ve',
    '--val_while_n_epoch',
    action='store',
    type=int,
    dest='ValWhileNEpoch',
    default=5,
    help='while epoch N * 5 end , would go into val epoch, default 5'
)
args = parser.parse_args()

if __name__ == '__main__':
    (options, args) = parser.parse_args()
    assert options.ValRate < 0.5, 'val rate should be smaller than 0.5'
    data_root = '../../Data/mojing3/'
    if options.Mode == 'train':
        sources = data_root + 'train.csv'
        train = True
        test = False
    else:
        sources = data_root + 'test.csv'
        train = False
        test = True
        pass
    print('Sources File: ', sources)
    print('Run Mode: ', options.Mode)
    print('Val Rate: ', options.ValRate)
    print('Batch Size: ', options.BatchSize)
    model = model.Model()
    data_handle = data.Data(sources=sources, batch_size=options.BatchSize, val_rate=options.ValRate, train=train, test=test)
    if train is True:
        print('epoch train batch size: ', data_handle.train_batch_num)
        print('epoch train batch size: ', data_handle.val_batch_num)
    if test is True:
        print('test data num: ', data_handle.test_data_num)
    if options.DataType == 'char':
        char = True
        word = False
        max_time_step = data_handle.char_fixed_length
    else:
        char = False
        word = True
        max_time_step = data_handle.word_fixed_length
    print('data type: ', options.DataType)
    print('max time step: ', max_time_step)
    hidden_unit = [int(i) for i in options.HiddenUnit.split(' ')]
    print('hidden unit: ', hidden_unit)
    print('Epoch: ', options.Epoch)
    print('display while {0} step'.format(options.DisplayWhileNStep))
    print('save while {0} step'.format(options.SaveWhileNStep))
    print('val while {0} epoch'.format(options.ValWhileNEpoch))
    model.build(embeding_len=embeding_len, batch_size=options.BatchSize, hidden_unit=hidden_unit,
                max_time_step=max_time_step)
    model.build_loss()
    model.train(epoch=options.Epoch, save_path=options.SavePath, save_while_n_step=options.SaveWhileNStep,
                val_while_n_epoch=options.ValWhileNEpoch,
                data=data_handle, char=char, word=word, display_shilw_n_step=options.DisplayWhileNStep,
                basic_lr = options.LearningRate, device=options.UseCpu)
    pass
