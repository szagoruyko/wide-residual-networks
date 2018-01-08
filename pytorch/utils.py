import torch
import torch.cuda.comm as comm
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from torch.autograd import Variable
from nested_dict import nested_dict
from collections import OrderedDict


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return cast(kaiming_normal(torch.Tensor(no, ni, k, k)))


def linear_params(ni, no):
    return cast({'weight': kaiming_normal(torch.Tensor(no, ni)), 'bias': torch.zeros(no)})


def bnparams(n):
    return cast({'weight': torch.rand(n), 'bias': torch.zeros(n)})


def bnstats(n):
    return cast({'running_mean': torch.zeros(n), 'running_var': torch.ones(n)})


def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, stats, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]
    stats_replicas = [dict(zip(stats.keys(), p))
                      for p in comm.broadcast_coalesced(list(stats.values()), device_ids)]

    replicas = [partial(f, params=p, stats=s, mode=mode)
                for p, s in zip(params_replicas, stats_replicas)]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten_params(params):
    return OrderedDict(('.'.join(k), Variable(v, requires_grad=True))
                       for k, v in nested_dict(params).iteritems_flat() if v is not None)


def flatten_stats(stats):
    return OrderedDict(('.'.join(k), v)
                       for k, v in nested_dict(stats).iteritems_flat())


def batch_norm(x, params, stats, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=stats[base + '.running_mean'],
                        running_var=stats[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3),
              str(tuple(v.size())).ljust(23), torch.typename(v.data if isinstance(v, Variable) else v))
