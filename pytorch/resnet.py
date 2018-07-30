import torch
import torch.nn.functional as F
import utils


def resnet(depth, width, num_classes, dropout_prob, activation_dropout):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def activation_dropout(x, p_drop, training):
        if training:
            # the input P. is the base DROPOUT PROBABILITY
            P = 1.-p_drop
            # x.size() = [bs, f , h, w]
            #sum over w and h to get total activation of a filter across space -> [bs, f]
            # normalize feature activations to 1 for each example in the batch
            bs, N, w, h = x.size()
            p_act = F.normalize(x.sum(-1).sum(-1), p=1, dim=-1)
            p_retain = 1. - ( (1.-P)*(N-1.)*p_act ) / ( ((1.-P)*N-1.)*p_act+P )
            mask = torch.bernoulli(p_retain)
            scale = mask.mean(-1)
            mask = mask/torch.stack([scale for i in range(N)], -1)
            mask =  torch.stack([mask for i in range(w)], -1)
            mask =  torch.stack([mask for i in range(h)], -1)
            
            return mask*x
        else:
            return x
        
    
    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        if activation_dropout:
            o2 = activation_dropout(o2, p_drop=dropout_prob, training=mode)
        elif dropout_prob:
            o2 = F.dropout2d(o2, p=dropout_prob, training=mode)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode):
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, 'group0', mode, 1)
        g1 = group(g0, params, 'group1', mode, 2)
        g2 = group(g1, params, 'group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params
