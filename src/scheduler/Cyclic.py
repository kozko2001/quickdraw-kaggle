import numpy as np
import os
import matplotlib.pyplot as plt
import math
from torch.optim import Optimizer

from torch.optim.lr_scheduler import _LRScheduler

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle.
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up.
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_idx (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """
    def __init__(self,
                 optimizer,
                 base_lr=1e-3,
                 max_lr=6e-3,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 last_batch_idx=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        base_lrs = self._format_lr('base_lr', optimizer, base_lr)
        if last_batch_idx == -1:
            for base_lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = base_lr
        self.max_lrs = self._format_lr('max_lr', optimizer, max_lr)
        step_size_down = step_size_down or step_size_up
        self.total_size = float(step_size_up + step_size_down)
        self.step_ratio = float(step_size_up) / self.total_size
        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        super(CyclicLR, self).__init__(optimizer, last_batch_idx)
    def _format_lr(self, name, optimizer, lr):
        """Return correctly formatted lr for each param group."""
        if isinstance(lr, (list, tuple)):
            if len(lr) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(lr)))
            return np.array(lr)
        else:
            return lr * np.ones(len(optimizer.param_groups))
    def _triangular_scale_fn(self, x):
        return 1.
    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))
    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)
    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.
        """
        cycle = np.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)
        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)
        return lrs

    def step_batch(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        self.last_epoch = self.last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        return lr


    def __call__(self, iter):
        return self.step_batch(iter)


    def step(self, v):
        pass


class CyclicScheduler():

    def __init__(self, optim, min_lr=0.001, max_lr=0.01, period=10, max_decay=0.99, warm_start=0 ):
        super(CyclicScheduler, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.max_decay = max_decay
        self.warm_start = warm_start
        self.cycle = -1
        self.time = 0
        self.optim = optim

    def __call__(self, time):
        if time<self.warm_start: return self.max_lr

        #cosine
        self.cycle = (time-self.warm_start)//self.period
        time = (time-self.warm_start)%self.period

        period = self.period
        min_lr = self.min_lr
        max_lr = self.max_lr *(self.max_decay**self.cycle)


        r   = (np.tanh(-time/period *16 +8)+1)*0.5
        lr = min_lr + r*(max_lr-min_lr)

        return lr



    def __str__(self):
        string = 'CyclicScheduler\n' \
                + 'min_lr=%0.3f, max_lr=%0.3f, period=%8.1f'%(self.min_lr, self.max_lr, self.period)
        return string


    def step(self, val):
        self.time = self.time + 1
        new_lr = self(self.time)
        adjust_learning_rate(self.optim, new_lr)



def plot_rates(fig, lrs, title=''):

    N = len(lrs)
    epoches = np.arange(0,N)


    #get limits
    max_lr  = np.max(lrs)
    xmin=0
    xmax=N
    dx=2

    ymin=0
    ymax=max_lr*1.2
    dy=(ymax-ymin)/10
    dy=10**math.ceil(math.log10(dy))

    ax = fig.add_subplot(111)
    #ax = fig.gca()
    ax.set_axisbelow(True)
    ax.set_xlim(xmin,xmax+0.0001)
    ax.set_ylim(ymin,ymax+0.0001)
    ax.grid(b=True, which='minor', color='black', alpha=0.1, linestyle='dashed')
    ax.grid(b=True, which='major', color='black', alpha=0.4, linestyle='dashed')

    ax.set_xlabel('iter')
    ax.set_ylabel('learning rate')
    ax.set_title(title)
    ax.plot(epoches, lrs)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    from torch import optim
    from torch import nn

    num_iters=60

    net = nn.Linear(10, 100)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))



    scheduler = CyclicLR(optimizer,step_size_up = 1000, mode="triangular2")
    num_iters = 12000

    lrs = np.zeros((num_iters),np.float32)
    for iter in range(num_iters):

        lr = scheduler(iter)
        lrs[iter] = lr
        if lr<0:
            num_iters = iter
            break
#        print ('iter=%02d,  lr=%f   '%(iter,lr))


    #plot
    fig = plt.figure()
    plot_rates(fig, lrs, title=str(scheduler))
    plt.show()
