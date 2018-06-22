# coding = utf-8
import model
import data
from optparse import OptionParser

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
    '--ss',
    '--save_while_n_step',
    action='store',
    type=int,
    dest='SaveWhileNStep',
    default=2,
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
    if options.Mode == 'train':
        sources = './Data/train.csv'
        train = True
        test = False
    else:
        sources = './Data/test.csv'
        train = False
        test = True
        pass
    print('Sources File: ', sources)
    print('Run Mode: ', options.Mode)
    print('Val Rate: ', options.ValRate)
    print('Batch Size: ', options.BatchSize)
    model = model.Model()
    data_handle = data.Data(sources=sources, batch_size=options.BatchSize, val_rate=options.ValRate, train=train, test=test)
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
    print('save while {0} step'.format(options.SaveWhileNStep))
    print('val while {0} epoch'.format(options.ValWhileNEpoch))
    model.build(embeding_len=300, batch_size=options.BatchSize, hidden_unit=hidden_unit,
                max_time_step=max_time_step)
    model.build_loss()
    model.train(epoch=options.Epoch, save_path='./check/', save_while_n_step=options.SaveWhileNStep,
                val_while_n_epoch=options.ValWhileNEpoch,
                data=data_handle, char=char, word=word)
    pass
