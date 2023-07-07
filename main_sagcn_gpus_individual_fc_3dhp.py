import glob
import logging
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from common.camera import get_uvd2xyz
from common.h36m_dataset import Human36mDataset
from common.load_data_3dhp_mae import Fusion
#from common.load_data_hm36 import Fusion
from common.opt import opts
from common.utils import *
from model.block.refine import refine
from model.s_agcn_3d_seq_individual_fc import S_AGCN as Model
import torch.distributed as dist
import warnings

opt = opts().parse()

dist.init_process_group(backend='nccl')

opt.batch_size = int(opt.batch_size /
                     torch.distributed.get_world_size())
opt.lr = opt.lr / torch.distributed.get_world_size()
warnings.filterwarnings("ignore")


def seed_everything(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(42)


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']

    if split == 'train':
        model_trans.train()
        model_refine.train()
    else:
        model_trans.eval()
        model_refine.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test =  AccumLoss()

    action_error_sum = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)

    # joints_left = [4, 5, 6, 11, 12, 13]
    # joints_right = [1, 2, 3, 14, 15, 16]
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]
    pbar = tqdm(total=len(dataLoader))

    if split == 'train':
        pbar.set_description("[TRAIN]")
    else:
        pbar.set_description("[TEST]")

    for i, data in enumerate(dataLoader):
        if opt.debug and i > 1:
            break

        #batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        if split == "train":
            batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        else:
            batch_cam, gt_3D, input_2D, action, scale, bb_box = data
        [input_2D, gt_3D, batch_cam, scale,
         bb_box] = get_varialbe(split,
                                [input_2D, gt_3D, batch_cam, scale, bb_box],
                                device=torch.device(f'cuda:{dist.get_rank()}'))

        N = input_2D.size(0)

        out_target = gt_3D.clone().reshape(N, -1, opt.out_joints, opt.out_channels)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.reshape(N, -1, opt.out_joints, opt.out_channels).float().to(
            torch.device(f'cuda:{dist.get_rank()}'))

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.test_augmentation and split == 'test':
            input_2D, output_3D, output_3D_VTE = input_augmentation(
                input_2D, model_trans, joints_left, joints_right)
        else:
            input_2D = input_2D.reshape(N, -1, opt.n_joints, opt.in_channels,
                                     1).permute(0, 3, 1, 2, 4).float().to(
                                         torch.device(f'cuda:{dist.get_rank()}'))

            output_3D, output_3D_VTE = model_trans(input_2D)

        output_3D_VTE = output_3D_VTE.permute(0, 2, 3, 4,
                                              1).reshape(N, -1, opt.out_joints,
                                                      opt.out_channels)
        output_3D = output_3D.permute(0, 2, 3, 4,
                                      1).reshape(N, -1, opt.out_joints,
                                              opt.out_channels)

        output_3D_VTE = output_3D_VTE * scale.unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1).repeat(1, output_3D_VTE.size(1), opt.out_joints,
                                     opt.out_channels)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1).repeat(1, output_3D.size(1), opt.out_joints, opt.out_channels)
        output_3D_single = output_3D
        if split == 'train':
            pred_out = output_3D_VTE

        elif split == 'test':
            pred_out = output_3D_single

        input_2D = input_2D.permute(0, 2, 3, 1, 4).reshape(N, -1, opt.n_joints, 2)

        loss = mpjpe_cal(pred_out, out_target) + mpjpe_cal(
            output_3D_single, out_target_single)

        if opt.refine:
            pred_uv = input_2D
            uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1),
                             output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            xyz[:, :, 0, :] = 0
            import pdb; pdb.set_trace()

            post_out = model_refine(output_3D_single, xyz)
            loss = mpjpe_cal(post_out, out_target_single) + 0.0 * loss
        else:
            loss = mpjpe_cal(pred_out, out_target) + mpjpe_cal(
                output_3D_single, out_target_single)

        running_loss = loss.item()
        loss_all['loss'].update(running_loss * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.refine:
                post_out[:, :, 14, :] = 0
                joint_error = mpjpe_cal(post_out, out_target_single).item()
            else:
                pred_out[:, :, 14, :] = 0
                joint_error = mpjpe_cal(pred_out, out_target).item()

            error_sum.update(joint_error * N, N)

        elif split == 'test':
            pred_out[:, :, 14, :] = 0
            #import pdb; pdb.set_trace()
            joint_error_test = mpjpe_cal(pred_out, out_target).item()#test_calculation(pred_out, out_target, action,
                                #                action_error_sum, opt.dataset,
                                #                subject)
            error_sum_test.update(joint_error_test * N, N)
            #if opt.refine:
            #    post_out[:, :, 14, :] = 0

            #    action_error_sum_post_out = mpjpe_cal(post_out, out_target_single).item()#test_calculation(
                    #post_out, out_target, action, action_error_sum_post_out,
                    #opt.dataset, subject)

        pbar.update(1)
        pbar.set_postfix(ordered_dict={'loss': running_loss})
    pbar.close()

    if split == 'train':
        return loss_all['loss'].avg, error_sum.avg #* 1000
    elif split == 'test':
        #if opt.refine:
        #    mpjpe = print_error(opt.dataset, action_error_sum_post_out,
        #                        opt.train)
        #else:
        #    mpjpe = print_error(opt.dataset, action_error_sum, opt.train)

        return error_sum_test.avg #* 1000 #mpjpe


def input_augmentation(input_2D, model_trans, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1].reshape(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].reshape(N, T, J, C,
                                            1).permute(0, 3, 1, 2, 4)

    #input_2D_flip = input_2D_flip.to(torch.device(f'cuda:{dist.get_rank()}'))
    output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip)

    output_3D_flip_VTE[:, 0] *= -1
    output_3D_flip[:, 0] *= -1

    output_3D_flip_VTE[:, :, :, joints_left +
                       joints_right] = output_3D_flip_VTE[:, :, :,
                                                          joints_right +
                                                          joints_left]
    output_3D_flip[:, :, :, joints_left +
                   joints_right] = output_3D_flip[:, :, :,
                                                  joints_right + joints_left]
    #input_2D_non_flip = input_2D_non_flip.to(torch.device(f'cuda:{dist.get_rank()}'))
    output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip)

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D, output_3D_VTE


if __name__ == '__main__':
    opt.manualSeed = 1

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    #dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions_3dhp(opt.actions)
    print(actions)

    if opt.train:
        train_data = Fusion(opt=opt,
                            train=True,
                            #dataset=dataset,
                            root_path=root_path)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=torch.distributed.get_world_size(), rank=dist.get_rank())
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=opt.batch_size,
            num_workers=int(opt.workers),
            pin_memory=True,
            sampler=train_sampler)
    if opt.test:
        test_data = Fusion(opt=opt,
                           train=False,
                           #dataset=dataset,
                           root_path=root_path)
        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size *
            torch.distributed.get_world_size(),
            shuffle=False,
            num_workers=int(opt.workers),
            pin_memory=True)

    opt.out_joints = 17 #dataset.skeleton().num_joints()

    model = {}
    if opt.frames == 27:
        architecture = '3,3,3'
    elif opt.frames == 81:
        architecture = '3,3,3,3'
    else:
        if opt.frames != 243:
            print(
                '[Warning] Please check the default architecture [3,3,3,3,3], which may cause errors.'
            )
        architecture = '3,3,3,3,3'
    filter_widths = [int(x) for x in architecture.split(',')]

    if opt.train:
        #print('dist.get_rank(): ',dist.get_rank())
        p_2d = list(train_data.generator.poses_2d.values())[0]
        p_3d = list(train_data.generator.poses_3d.values())[0]
        model['trans'] = Model(p_2d.shape[-2],
                               p_2d.shape[-1],
                               p_3d.shape[-2],
                               filter_widths=filter_widths,
                               causal=False,
                               dropout=0.1,
                               channels=opt.channel,
                               dataset='h36m').to(
                                   torch.device(f'cuda:{dist.get_rank()}'))
    else:
        p_2d = list(test_data.generator.poses_2d.values())[0]
        p_3d = list(test_data.generator.poses_3d.values())[0]
        model['trans'] = Model(p_2d.shape[-2],
                               p_2d.shape[-1],
                               p_3d.shape[-2],
                               filter_widths=filter_widths,
                               causal=False,
                               dropout=0.1,
                               channels=opt.channel,
                               dataset='h36m').to(
                                   torch.device(f'cuda:{dist.get_rank()}'))

    #model['trans'] = Model(opt).to(torch.device(f'cuda:{dist.get_rank()}'))
    model['refine'] = refine(opt).to(torch.device(f'cuda:{dist.get_rank()}'))

    for i_model in model:
        model_params=0
        for parameter in model[i_model].parameters():
            model_params += parameter.numel()
        print('INFO: Number of parameters in model {} count: {:.2f}M'.format(i_model, model_params/1000000))

    model['trans'] = nn.parallel.DistributedDataParallel(
        model['trans'], device_ids=[dist.get_rank()])
    model['refine'] = nn.parallel.DistributedDataParallel(
        model['refine'], device_ids=[dist.get_rank()])
    # https://stackoverflow.com/questions/50442000/dataparallel-object-has-no-attribute-init-hidden/51377405
    if isinstance(model['trans'], nn.parallel.DistributedDataParallel):
        model_train_attr_accessor = model['trans'].module
    if isinstance(model['refine'], nn.parallel.DistributedDataParallel):
        model_refine_train_attr_accessor = model['refine'].module

    model_dict = model['trans'].state_dict()
    if opt.reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        refine_path = []
        for path in model_path:
            if path.split('/')[-1][0] == 'n':
                no_refine_path = path
                print(no_refine_path)
                break

        pre_dict = torch.load(no_refine_path, map_location=torch.device(f'cuda:{dist.get_rank()}'))#torch.cuda.current_device())
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['trans'].load_state_dict(model_dict)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        refine_path = []
        for path in model_path:
            if path.split('/')[-1][0] == 'r':
                refine_path = path
                print(refine_path)
                break

        pre_dict_refine = torch.load(refine_path, map_location=torch.device(f'cuda:{dist.get_rank()}'))
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)


    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train:
            train_sampler.set_epoch(epoch)
            loss, error = train(opt, actions, train_dataloader, model,
                                optimizer_all, epoch)

        if opt.test and dist.get_rank() == 0:
            #import pdb; pdb.set_trace()
            mpjpe = val(
                opt, actions, test_dataloader, {
                    'trans':
                    nn.DataParallel(model['trans'].module,
                                    device_ids=list(
                                        range(torch.cuda.device_count())),
                                    output_device=torch.cuda.current_device()),
                    'refine':
                    nn.DataParallel(model['refine'].module,
                                    device_ids=list(
                                        range(torch.cuda.device_count())),
                                    output_device=torch.cuda.current_device())
                })
            mpjpe = mpjpe#[0]
            data_threshold = mpjpe#[0] # use the p#1
            #import pdb; pdb.set_trace()
            if opt.train and data_threshold < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name,
                                               opt.checkpoint, epoch,
                                               data_threshold, model['trans'],
                                               'no_refine')

                if opt.refine:
                    opt.previous_refine_name = save_model(
                        opt.previous_refine_name, opt.checkpoint, epoch,
                        data_threshold, model['refine'], 'refine')
                opt.previous_best_threshold = data_threshold

            print('mpjpe: %.2f' % (mpjpe))

            if not opt.train:
                break
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, mpjpe: %.2f' %
                             (epoch, lr, loss, mpjpe))
                print('e: %d, lr: %.7f, loss: %.4f, mpjpe: %.2f' %
                      (epoch, lr, loss, mpjpe))
        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay
