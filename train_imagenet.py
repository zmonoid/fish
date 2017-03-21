import mxnet as mx
import argparse
import os
import train_model

# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(
    description='train an image classifer on imagenet')
parser.add_argument(
    '--network',
    type=str,
    default='inception-bn',
    choices=[
        'alexnet', 'googlenet', 'inception-bn', 'inception-v3', 'resnet-50',
        'resnet-101', 'vgg'
    ],
    help='the cnn to use')
parser.add_argument(
    '--aug-level',
    type=int,
    default=3,
    choices=[1, 2, 3],
    help='level 1: use only random crop and random mirror\n'
    'level 2: add scale/aspect/hsv augmentation based on level 1\n'
    'level 3: add rotation/shear augmentation based on level 2')
parser.add_argument(
    '--data-dir', type=str, required=True, help='the input data directory')
parser.add_argument(
    '--model-prefix', type=str, help='the prefix of the model to load')
parser.add_argument(
    '--save-model-prefix', type=str, help='the prefix of the model to save')
parser.add_argument(
    '--lr', type=float, default=.01, help='the initial learning rate')
parser.add_argument(
    '--lr-factor',
    type=float,
    default=1,
    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument(
    '--lr-factor-epoch',
    type=float,
    default=1,
    help='the number of epoch to factor the lr, could be .5')
parser.add_argument(
    '--clip-gradient',
    type=float,
    default=5.,
    help='clip min/max gradient to prevent extreme value')
parser.add_argument(
    '--num-epochs', type=int, default=20, help='the number of training epochs')
parser.add_argument(
    '--load-epoch',
    type=int,
    help="load the model on an epoch using the model-prefix")
parser.add_argument(
    '--batch-size', type=int, default=32, help='the batch size')
parser.add_argument(
    '--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument(
    '--kv-store', type=str, default='local', help='the kvstore type')
parser.add_argument(
    '--num-examples',
    type=int,
    default=1281167,
    help='the number of training examples')
parser.add_argument(
    '--num-classes', type=int, default=1000, help='the number of classes')
parser.add_argument('--log-file', type=str, help='the name of log file')
parser.add_argument(
    '--log-dir', type=str, default="/tmp/", help='directory of the log file')
parser.add_argument(
    '--train-dataset',
    type=str,
    default="train.rec",
    help='train dataset name')
parser.add_argument(
    '--val-dataset',
    type=str,
    default="val.rec",
    help="validation dataset name")
parser.add_argument(
    '--data-shape', type=int, default=224, help='set image\'s shape')
parser.add_argument(
    '--finetune', type=bool, default=False, help='whether load to finetune')
parser.add_argument(
    '--finetune-scale',
    type=float,
    default=10,
    help='whether load to finetune')

args = parser.parse_args()
# network
import importlib
if args.network.startswith('resnet'):
    name = args.network.split('-')[0]
    depth = args.network.split('-')[1]
    net = importlib.import_module('symbols.' + name).get_symbol(
        args.num_classes, int(depth), '3, 224, 224')
else:
    net = importlib.import_module('symbols.' + args.network).get_symbol(
        args.num_classes)


# data
def get_iterator(args, kv):
    data_shape = (3, args.data_shape, args.data_shape)
    train = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.data_dir, args.train_dataset),
        #mean_r=123.68,
        #mean_g=116.779,
        #mean_b=103.939,
        data_shape=data_shape,
        batch_size=args.batch_size,
        #mean_img    = os.path.join(args.data_dir, "mean_%d.bin" % args.data_shape),
        pad=0,
        fill=127,
        #max_random_scale=1.0,  # 480 with imagnet, 32 with cifar10
        #min_random_scale=1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio=0 if args.aug_level == 1 else 0.25,
        random_h=0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s=0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l=0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle=0 if args.aug_level <= 2 else 10,
        max_shear_ratio=0 if args.aug_level <= 2 else 0.1,
        rand_mirror=True,
        rand_crop=True,
        shuffle=True,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.data_dir, args.val_dataset),
        #mean_r=123.68,
        #mean_g=116.779,
        #mean_b=103.939,
        #label_name  = 'svm_label',
        #rand_crop   = False,
        rand_mirror=False,
        data_shape=data_shape,
        batch_size=args.batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank)
    #train.rename_label = ['svm_label']
    #val.rename_label = ['svm_label']

    return (train, val)


# train
train_model.fit(args, net, get_iterator)
