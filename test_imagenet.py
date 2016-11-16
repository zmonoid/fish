import mxnet as mx
import numpy as np
import argparse
#from multiprocessing import Pool


parser = argparse.ArgumentParser(description='train an image classifer on imagenet')

parser.add_argument('--load-epoch', type=int, default=1,
                    help='the prefix of the model to load')
parser.add_argument('--load-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--gpus', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--shape', type=int,
                    help='the prefix of the model to load')
parser.add_argument('--batch-size', type=int,
                    help='the prefix of the model to load')
parser.add_argument('--data-shape', type=int,
                    help='the prefix of the model to load')
args = parser.parse_args()


model = mx.model.FeedForward.load(args.load_prefix, args.load_epoch)
ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
test = mx.io.ImageRecordIter(
        path_imgrec = "./data/test_%d.rec" % args.data_shape,
        #mean_img    = "./data/mean_%d.bin" % args.shape,
        data_shape  = (3, args.shape, args.shape),
        batch_size  = args.batch_size,
        mean_r      = 123.68,
        mean_g      = 116.779,
        mean_b      = 103.939,
        random_crop     = True,
        #random_mirror   = True,
        shuffle = False)

prob_mod = mx.model.FeedForward(
        ctx = ctx,
        symbol = model.symbol,
        arg_params = model.arg_params,
        aux_params = model.aux_params,
        allow_extra_params = True)


prob = prob_mod.predict(test)
for _ in range(5):
    test.reset()
    print _
    prob += prob_mod.predict(test)
prob /= 6
print prob.shape
np.save('./data/out.npy', prob)
