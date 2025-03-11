import argparse
import os
import torch
from exp.exp_open_net import Exp_Main_DualmodE3K
from exp.exp_open_net_FRnet import Exp_Main_DualmodE3K_FRnet
from exp.opt_urt import Opt_URT
from exp.opt_urt_FRNet import Opt_URT_FRNet

import warnings
import random
import numpy as np

import gc

warnings.filterwarnings('ignore')

def main():    
    parser = argparse.ArgumentParser(description='iTransformer for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='urt3_96_96_Exchange', help='model id')
    parser.add_argument('--model', type=str, required=False, default='B6iFast',
                        help='model name, options: [Autoformer, Informer, Transformer]')
    parser.add_argument('--slow_model', type=str, required=False, default='S1iSlow',
                        help='slow model name, options: [Autoformer, Informer, Transformer, etc]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/exchange_rate', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='exchange_rate.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_learner', type=int, default=3, help='number of learner')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--urt_heads', type=int, default=3, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--anomaly', type=float, default=10.0, help='anomaly limit')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--corr_penalty', type=float, default=0.5, help='correlation penalty for negative correlation loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_aclr', action='store_true', help='use adaptive cyclic learning rate', default=False)
    
    # Slow Learner
    parser.add_argument('--use_slow_learner', type=int, default=1, help='use slow learner')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=96, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # FRNet
    parser.add_argument('--pred_head_type', type=str, default='linear', help='linear or truncation')
    parser.add_argument('--aggregation_type', type=str, default='linear', help='linear or avg')
    parser.add_argument('--channel_attention', type=int, default=0, help='True 1 or False 0')
    parser.add_argument('--global_freq_pred', type=int, default=1, help='True 1 or False 0')
    parser.add_argument('--period_list', type=int, nargs='+', default=1, help='period_list') 
    parser.add_argument('--emb', type=int, default=64, help='patch embedding size')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--fix_seed', type=str, default='2021,2022,2023', help='Fix seed for iterations')

    # parser.add_argument('--num_fastlearners', type=int, default=2, help='number of fast_learner')


    args = parser.parse_args()

    # fix_seed = 2021
    # fix_seed=args.fix_seed.split(",")
    fix_seed=int(args.fix_seed.split(",")[0])
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        # args.gpu = args.device_ids
        # args.gpu = args.devices
        print("args.gpu: ")
        print(args.gpu)

    print('Args in experiment:')
    print(args)

    
    # Exp = Exp_Main_Dualmod
    # Exp = Exp_Main_DualmodE3K
    Exp = Exp_Main_DualmodE3K_FRnet

    if args.is_training:
        for ii in range(args.itr):
            
            fix_seed=args.fix_seed.split(",")
            fix_seed=[int(i) for i in fix_seed]
            random.seed(fix_seed[ii])
            torch.manual_seed(fix_seed[ii])
            np.random.seed(fix_seed[ii])
            torch.cuda.manual_seed(fix_seed[ii])
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(fix_seed[ii])

            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.d_layers,
                args.des)

            exp = Exp(args)  # set experiments 0
            # opt = OptURT(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # print('>>>>>>>start training URT: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # opt.train_urt(setting)

            # Testing only Mantra
            print('>>>>>>>testing only mantra : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting) 

            # print('>>>>>>>set rl data : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.set_rl_data(setting)

            gc.collect()
            torch.cuda.empty_cache()

            # RL Experiment
            # print('>>>>>>>train RL : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.train_rl(setting)

            # print('>>>>>>>testing Model+URT : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # opt.test2(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            del exp
            gc.collect()
            torch.cuda.empty_cache() 

        # OptRL = OPT_RL_Mantra(args)
        # for ii in range(args.itr):
        # OptURT = Opt_URT -> ini dipake
        # OptURT = Opt_URT_FRNet

        # for ii in range(args.itr): -> sini
        #     setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_{}'.format(
        #         args.model_id,
        #         args.model,
        #         args.data,
        #         args.features,
        #         args.seq_len,
        #         args.label_len,
        #         args.pred_len,
        #         args.d_model,
        #         args.e_layers,
        #         args.d_layers,
        #         args.des)
            
        #     opt = OptURT(args)  # set experiments

        #     print('>>>>>>>start training URT: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        #     opt.train_urt(setting)

        #     print('>>>>>>>testing FastSlow+URT : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     opt.test2(setting)

        #     gc.collect()
        #     torch.cuda.empty_cache() -> sampe sini
        #     setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_{}'.format(
        #         args.model_id,
        #         args.model,
        #         args.data,
        #         args.features,
        #         args.seq_len,
        #         args.label_len,
        #         args.pred_len,
        #         args.d_model,
        #         args.e_layers,
        #         args.d_layers,
        #         args.des)

        #     # exp = Exp(args)  # set experiments
        #     # opt = OptURT(args)  # set experiments

        #     # print('>>>>>>>start training URT: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        #     # opt.train_urt(setting)

        #     # print('>>>>>>>testing FastSlow+URT : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     # opt.test2(setting)

        #     torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_{}'.format(args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.des)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()