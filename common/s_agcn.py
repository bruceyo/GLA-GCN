# Copyright (c) 2021-present, The Hong Kong Polytechnic University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from .agcn import Graph, TCN_GCN_unit

import numpy as np

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels, dataset):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout, inplace=True)

        self.pad = [ filter_widths[0] // 2 ]

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad[:len(self.filter_widths)]:
            frames += f
        return 1 + 2*frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x

class S_AGCN(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=96, dataset='h36m'):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of graph convolution channels
        dataset -- default is h36m, can be set to humaneva15
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, dataset)

        layers_tcngcn = []
        num_person = 1
        in_channels = 2
        num_point = num_joints_in
        self.graph = Graph(dataset)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        A = self.graph.A
        self.expand_gcn = TCN_GCN_unit(2, channels, A)

        self.causal_shift = []
        next_dilation = filter_widths[0]
        for i in range(0, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers_tcngcn.append(TCN_GCN_unit(channels, channels, A))
            layers_tcngcn.append(TCN_GCN_unit(channels, channels, A, stride=filter_widths[i], residual=False))
            next_dilation *= filter_widths[i]

        self.layers_tcngcn = nn.ModuleList(layers_tcngcn)
        self.fc = nn.Conv1d(channels, 3, 1)

    def set_bn_momentum(self, momentum):
        self.data_bn.momentum = momentum
        self.expand_gcn.gcn1.bn.momentum = momentum
        self.expand_gcn.tcn1.bn.momentum = momentum
        for layer in self.layers_tcngcn:
            layer.gcn1.bn.momentum = momentum
            layer.tcn1.bn.momentum = momentum

    def _forward_blocks(self, x):

        N, V, T = x.size()
        v = V//2 # number of 2D pose joints
        x = self.data_bn(x)
        x = x.view(N, 1, v, 2, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N, 2, T, v)

        x = self.expand_gcn(x)
        for i in range( len(self.pad) -1):
            res = x[:, :, self.causal_shift[i] + self.filter_widths[i]//2 :: self.filter_widths[i], :]

            x = self.drop(self.layers_tcngcn[2*i](x))
            x = self.drop(self.layers_tcngcn[2*i+1](x))
            x = res + x
        pose_3d_ = x

        pose_3d = torch.from_numpy(np.full((N, 3, v),0).astype('float32')).cuda(x.get_device())
        for i in range(0,v):
            pose_joint_3d = pose_3d_[:,:,:,i].mean(2)
            pose_joint_3d = self.fc(pose_joint_3d.view(N,-1,1))
            pose_3d[:,:,i] = pose_joint_3d.view(N,-1)

        return pose_3d
