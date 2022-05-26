"""Utility functions
Some utility functions inlude:
    - infer_memory_size: estimate memory size for a dnn model
    - svd_approx: approximated SVD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

dim_prev = []

def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def infer_memory_size(sgx_dnn, model, need_sgx, batchsize, dim_in):
    """
    Infer memory requirement in each layer in SGX
    :param sgx_dnn: class sgxDNN
    :param model: model definition (Asymmetric version)
    :param need_sgx: whether need SGX
    :param batchsize: batch size
    :param dim_in: dimension of input
    :return:
    """
    global dim_prev
    sgx_dnn.batchsize = batchsize
    sgx_dnn.dim_in = dim_in
    dim_block_first = dim_prev[:]
    for module, need in zip(model, need_sgx):
        if module.type == "asymResBlock":
            infer_memory_size( sgx_dnn, module.children(), need, batchsize, dim_in )
        if module.type == "asymShortCut":
            in_dim_cur = dim_prev
            out_dim_cur = in_dim_cur
            sgx_dnn.in_memory_desc[ module ]  = in_dim_cur
            sgx_dnn.out_memory_desc[ module ] = out_dim_cur
            sgx_dnn.lyr_config.append( None )
        if module.type == "asymReLU":
            in_dim_cur = dim_prev
            out_dim_cur = in_dim_cur
            sgx_dnn.in_memory_desc[module] = in_dim_cur
            sgx_dnn.out_memory_desc[module] = out_dim_cur
            sgx_dnn.lyr_config.append(None)
        elif module.type == 'asymConv2D' or isinstance(module, torch.nn.Conv2d):
            padding, dilation, kern_sz, stride = module.padding, module.dilation, module.kernel_size, module.stride
            if len(dim_prev) == 0:
                Hi, Wi = dim_in[1], dim_in[2]
            elif stride[ 0 ] == 1:
                Hi, Wi = dim_prev[2], dim_prev[3]
            else:
                Hi, Wi = dim_block_first[ 2 ], dim_block_first[ 3 ]
            Ho = np.floor((Hi + 2*padding[0] - dilation[0]*(kern_sz[0]-1) - 1) / stride[0] + 1)
            Wo = np.floor((Wi + 2*padding[1] - dilation[1]*(kern_sz[1]-1) - 1) / stride[1] + 1)
            Ho, Wo = int(Ho), int(Wo)
            in_dim_cur = [batchsize, module.in_channels, Hi, Wi]
            sgx_dnn.in_memory_desc[module] = in_dim_cur
            out_dim_cur = [batchsize, module.out_channels, Ho, Wo]
            sgx_dnn.out_memory_desc[module] = out_dim_cur
            if module.type == 'asymConv2D':
                sgx_dnn.lyr_config.append([module.in_channels, module.out_channels, kern_sz, stride, padding])
        elif module.type == "asymReLUPooling" or isinstance(module, torch.nn.MaxPool2d) or isinstance( module, torch.nn.AvgPool2d):
            padding, kern_sz, stride = module.padding, module.kernel_size, module.stride
            out_channels, Hi, Wi = dim_prev[1], dim_prev[2], dim_prev[3]
            Ho = np.floor((Hi + 2*padding - kern_sz) / stride + 1)
            Wo = np.floor((Wi + 2*padding - kern_sz) / stride + 1)
            Ho, Wo = int(Ho), int(Wo)
            in_dim_cur = [batchsize, out_channels, Hi, Wi]
            sgx_dnn.in_memory_desc[module] = in_dim_cur
            out_dim_cur = [batchsize, out_channels, Ho, Wo]
            sgx_dnn.out_memory_desc[module] = out_dim_cur
            sgx_dnn.lyr_config.append([out_channels, kern_sz, stride, padding])

        if module.type != "asymResBlock":
            dim_prev = out_dim_cur


def init_model(model):
    """
    custom weight init function for model in SGX.
    """
    for m in model:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def init_netGD( m ):
    """
    custom weight init function for generator
    """
    classname = m.__class__.__name__
    if classname.find( 'Conv' ) != -1:
        nn.init.normal_( m.weight.data, 0.0, 0.02 )
    elif classname.find( 'BatchNorm' ) != -1:
        nn.init.normal_( m.weight.data, 1.0, 0.02 )
        nn.init.constant_( m.bias.data, 0 )


def svd_approx( data, iters=2, maxinfo=0.1, eps=1e-6 ):
    """
    Approximated SVD
    :param data data to be decomposed (batch, channel, height*weight)
    :param r rank (int)
    :param iters number of iteration needed (int)
    :param maxinfo max information kept in untrusted platform
    :param eps
    :return low-rank output
    """
    b, c, hw = data.shape
    data_copy = data.clone()
    data_lowrank = torch.zeros_like( data )
    r_actual = 0
    energy_data = torch.norm(data) ** 2
    while True:
        u_i = torch.randn( b, c, 1 ) #.cuda()
        u_i = torch.nn.functional.normalize( u_i, dim=1 )
        # v_i = torch.randn( b, hw, 1 ).cuda()
        v_i = torch.mean( data_copy, dim=1, keepdim=True ).transpose( 1, 2 )
        # v_i = torch.nn.functional.normalize( v_i, dim=2 )

        # Alternate optimization
        for j in range( iters ):
            u_i = torch.div(
                torch.matmul( data_copy, v_i ),
                torch.matmul( v_i.transpose( 1, 2 ), v_i ) + eps
            )

            v_i = torch.div(
                torch.matmul(data_copy.transpose(1, 2), u_i),
                torch.matmul(u_i.transpose(1, 2), u_i) + eps
            )
        data_rank1 = torch.matmul( u_i, v_i.transpose( 1, 2 ) )
        data_lowrank += data_rank1
        data_copy -= data_rank1
        r_actual += 1

        # adaptively check if low-rank part is accurate enough
        energy_residual = torch.norm( data_copy )**2
        if ( energy_residual / energy_data ) <= maxinfo or r_actual == c:
            break

    del data_copy
    return data_lowrank, r_actual


def label_to_onehot( target, num_classes=10 ):
    target = torch.unsqueeze( target, 1 )
    onehot_target = torch.zeros( target.size(0), num_classes, device=target.device )
    onehot_target.scatter_( 1, target, 1 )
    return onehot_target


def cross_entropy_for_onehot( pred, target ):
    return torch.mean( torch.sum(- target * F.log_softmax(pred, dim=-1), 1) )
