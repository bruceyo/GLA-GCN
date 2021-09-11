# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
#from tqdm import tqdm

bone_pairs = [(1,0), (2,1), (3,2), (4,0), (5,4), (6,5), (7,0), (8,7),
            (9,8), (10,9), (11,8), (12,11), (13,12), (14,8), (15,14), (16,15)]

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def mpjpe_noise(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    target_pos = list(range(0,17))
    target_pos = np.random.choice(target_pos)
    #target_pos_2 = np.random.choice(target_pos)
    #print(target_pos)
    target[:,:,target_pos,:] = 0
    #target[target_pos[1]]=0
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def mpble(predicted, target):
    """
    Mean per-bone length error (i.e. mean Euclidean distance),
    """
    N, _, V, T = predicted.size()
    bone_length_loss = []
    for (v1, v2) in bone_pairs:
        predicted_bone = predicted[:,:,v1,:] - predicted[:,:,v2,:]
        predicted_bone_len = (predicted_bone.view(N,-1)*predicted_bone.view(N,-1)).sum(dim=1)**0.5
        target_bone = target[:,:,v1,:] - target[:,:,v2,:]
        target_bone_len = (target_bone.view(N,-1)*target_bone.view(N,-1)).sum(dim=1)**0.5
        #print('predicted_bone.size()',target_bone_len.size())
        #print('predicted_bone',target_bone_len)
        bone_length_loss.append(torch.mean(torch.norm(predicted_bone_len - target_bone_len, dim=len(target_bone_len.shape)-1)))
    #print()
    return 0.025 * sum(bone_length_loss)/16

def mpbde(predicted, target):
    """
    Mean per-bone direction error (i.e. mean ( 1- cos_theta) distance),
    """
    N, _, V, T = predicted.size()
    bone_direction_loss = []
    for (v1, v2) in bone_pairs:
        predicted_bone = predicted[:,:,v1,:] - predicted[:,:,v2,:]
        predicted_bone_len = (predicted_bone.view(N,-1)*predicted_bone.view(N,-1)).sum(dim=1)**0.5
        target_bone = target[:,:,v1,:] - target[:,:,v2,:]
        target_bone_len = (target_bone.view(N,-1)*target_bone.view(N,-1)).sum(dim=1)**0.5

        dot_product = (predicted_bone.view(N,-1)*target_bone.view(N,-1)).sum(dim=1)
        cos_theda = dot_product/(predicted_bone_len*target_bone_len)

        bone_direction_loss.append(torch.mean((1 - cos_theda), dim=len(cos_theda.shape)-1))

    return 0.5 * sum(bone_direction_loss)/16

def mpjpe_(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    #print('predicted.shape: ',predicted.shape) # predicted.shape:  torch.Size([64, 1, 17, 3])
    N, _, V, T = predicted.size()
    predicted_bone = torch.from_numpy(np.full((N, 1, 17, 3),0).astype('float32')).cuda(2)
    target_bone = torch.from_numpy(np.full((N, 1, 17, 3),0).astype('float32')).cuda(2)
    for (v1, v2) in bone_pairs:
        predicted_bone[:,:,v1,:] = predicted[:,:,v1,:] - predicted[:,:,v2,:]
        target_bone[:,:,v1,:] = target[:,:,v1,:] - target[:,:,v2,:]

    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1)),\
        torch.mean(torch.norm(predicted_bone - target_bone, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))

def mpjme(predicted_seq, target_seq, interval_set = [12]): # [N, 27, 17, 3]
    """
    Mean per-joint motion error (i.e. mean Euclidean distance), Abalations: multiple intervals, individual interval
    """
    assert predicted_seq.shape == target_seq.shape
    frames_len = predicted_seq.size()[1]
    loss_mpjme = 0
    for interval in interval_set:
        if interval > (frame_len -1) /2:
            continue
        for i in range( int((frames_len/2 + 1)/interval) ): # interval boundary
            #forward
            target_f = target_seq[:,frames_len/2+1 + interval,:,:] - target_seq[:,frames_len/2+1,:,:]
            predicted_f = predicted_seq[:,frames_len/2+1 + interval,:,:] - predicted_seq[:,frames_len/2+1,:,:]
            #backword
            target_b = target_seq[:,frames_len/2+1,:,:] - target_seq[:,frames_len/2+1 - interval,:,:]
            predicted_b = predicted_seq[:,frames_len/2+1,:,:] - predicted_seq[:,frames_len/2+1 - interval,:,:]
            loss_mpjme = loss_mpjme + torch.mean(torch.norm(predicted_f - target_f, dim=len(target_f.shape)-1)) + \
                                      torch.mean(torch.norm(predicted_b - target_b, dim=len(target_b.shape)-1))

    return loss_mpjme

def mpjse(predicted_seq, target_seq, interval_set = [12]):
    """
    Mean per-joint speed error (i.e. mean Euclidean distance),
    """
    assert predicted_seq.shape == target_seq.shape
