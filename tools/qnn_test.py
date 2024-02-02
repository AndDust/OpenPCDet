import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)

import torch.nn as nn
import copy


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="./cfgs/kitti_models/pointrcnn.yaml", help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default="../pointrcnn_7870.pth", help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    # TODO 新增量化参数
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')

    parser.add_argument('--init_wmode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for activation')
    parser.add_argument('--prob', default=0.5, type=float)

    parser.add_argument('--rand_seed', type=int, default=1024, help='random seed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    """ 固定随机种子 """
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.backends.cudnn.deterministic = True

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def get_qnn_model(args, PointRCNN_model):
    # TODO 加载模型
    PointRCNN_model.cuda()  # 将模型移动到GPU上
    PointRCNN_model.eval()  # 设置模型为评估模式

    fp_model = copy.deepcopy(PointRCNN_model)  # 深度复制模型
    fp_model.cuda()  # 将复制的模型移动到GPU上
    fp_model.eval()  # 设置复制的模型为评估模式

    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': args.init_amode,
                 'leaf_param': True, 'prob': args.prob}

    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()
    fp_model.eval()
    """fp_model只是用来对比，不需要开启量化"""
    fp_model.set_quant_state(False, False)  # 关闭量化状态

    """
       qnn是经过BN fold和开启量化状态的 （is_fusing=True）
    """
    qnn = QuantModel(model=PointRCNN_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    qnn.cuda()
    qnn.eval()

    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    """禁用神经网络中最后一个量化模块（QuantModule）的激活量化。"""
    qnn.disable_network_output_quantization()

    """得到量化后的模型"""
    print('the quantized model is below!')
    print(qnn)

    # PointRCNN_dataloader = create_PointRCNN_dataloader(logger)
    # pointnet_cali_data = get_train_samples(PointRCNN_dataloader, num_samples=args.pointRCNN_num_samples)

    # cali_data, cali_target = get_rain_samples(train_loader, num_samples=args.num_samples)
    # device = next(qnn.parameters()).device

    # Kwargs for weight rounding calibration
    """
        用于权重舍入校准的Kwargs
    """
    # kwargs = dict(cali_data=pointnet_cali_data, iters=args.iters_w, weight=args.weight,
    #               b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
    #               lr=args.lr, input_prob=args.input_prob, keep_gpu=not args.keep_cpu,
    #               lamb_r=args.lamb_r, T=args.T, bn_lr=args.bn_lr, lamb_c=args.lamb_c, a_count=args.a_count)

    set_weight_quantize_params(qnn)

    """
        重建: 重建就是让量化模型和FP模型的输出尽量保持一致,对量化模型的算子进行了重建,因为直接量化性能下降很多
    """
    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            """
                传入完整的qnn、fp_model和当前的module、fp_module
            """
            # layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
            layer_reconstruction(module)
        elif isinstance(module, BaseQuantBlock):
            # block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
            block_reconstruction(module)
        else:
            raise NotImplementedError

    """
        区块重建。对于第一层和最后一层，我们只能应用层重建。
    """

    # a_count = 0
    def recon_model(model: nn.Module, fp_model: nn.Module):
        """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp_model.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
                """比如对于ResNet里的包含conv BN Relu的block，在block层面再做一次"""
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
                recon_model(module, fp_module)

    """
        开始校准
    """
    # Start calibration
    recon_model(qnn, fp_model)
    #
    """qnn设置量化状态为True"""
    qnn.set_quant_state(weight_quant=True, act_quant=True)

    return qnn


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test,
    #                             pre_trained_path=args.pretrained_model)
    # model.cuda()
    
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test,
                                pre_trained_path=args.pretrained_model)

    qnn = get_qnn_model(args, model)

    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(qnn, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(qnn, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
