import mxnet as mx
from mxnet import gluon
import gluoncv as gcv

from ...scheduler.resource import get_gpu_count

def default_reward_fn(metric, net):
    return metric * ((net.avg_latency / net.latency) ** 0.07)

def init_default_train_args(batch_size, net, epochs, iters_per_epoch):
    base_lr = 0.1 * batch_size / 256
    lr_scheduler = gcv.utils.LRScheduler('cosine', base_lr=base_lr, target_lr=0.0001,
                                         nepochs=epochs, iters_per_epoch=iters_per_epoch)
    optimizer_params = {'wd': 1e-4, 'momentum': 0.9, 'lr_scheduler': lr_scheduler}
    train_args = {
        'trainer': gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)
    }

    train_args['batch_size'] = batch_size
    train_args['criterion'] = gluon.loss.SoftmaxCrossEntropyLoss()
    return train_args
