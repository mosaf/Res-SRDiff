
import argparse
import os
import sys
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as udata

import utils
from build import Model
from datasets.load_data import MotionCorruptedMRIDataset2D as ImageDataset
from torch.utils.data import DataLoader
FLAGS = None


def main():
    print(FLAGS.message)
    all_train_gt = '/path/to/train/data'
    all_valid_gt = "/path/to/valid/data"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    num_gpus = torch.cuda.device_count()

    if FLAGS.train:
        train_datset = ImageDataset(gt_name=all_train_gt)

        print(f"Loading data ...")

        train_dataloader = DataLoader(train_datset,
                                      batch_size=FLAGS.batch_size // num_gpus,
                                      shuffle=True,
                                      num_workers=min(FLAGS.train_num_workers, 4),
                                      drop_last=True,
                                      prefetch_factor= FLAGS.prefetch_factor,
                                      pin_memory=True)

        valid_datset = ImageDataset(gt_name=all_valid_gt)

        valid_dataloader = DataLoader(valid_datset,
                                      batch_size=FLAGS.test_batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=True)

        print('Creating model...\n')
        model = Model(name='Super_resolution',
                      device=device,
                      data_loader=train_dataloader,
                      test_data_loader=valid_dataloader,
                      FLAGS=FLAGS)

        model.train(epochs=FLAGS.epochs)


    else:

        all_test = "/path/to/test/data"
        test_datset = ImageDataset(gt_name=all_test, to_test=True)

        test_dataloader = DataLoader(test_datset,
                                      batch_size=FLAGS.test_batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)

        model = Model(name = FLAGS.model,
                      device=device,
                      data_loader=None,
                      test_data_loader=test_dataloader,
                      FLAGS=FLAGS)

        print('Loading Model')
        model.load_model_inference(path=FLAGS.out_dir)
        print('Evaluating Model')
        model.inference_(batch_size=FLAGS.test_batch_size, out_dir=FLAGS.out_dir_test)
        print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNetSwinTransformer')
    parser.add_argument('--message', type=str, default='Efficient SR-ResDiff model for LR --> HR', help='Diffusion model')
    parser.add_argument('--model', type=str, default='Super_resolution', help='Diffusion model')
    parser.add_argument('--model_name', type=str, default='unet_swuin', help='Diffusion model')
    parser.add_argument('--region', type=str, default='prostate', help='brain, prostate')

    parser.add_argument('--cuda', type=utils.boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPU used.')
    parser.add_argument('--train', type=utils.boolean_string, default=True, help='train mode or eval mode.')
    parser.add_argument('--resume', type=utils.boolean_string, default=False, help='train mode or eval mode.')

    parser.add_argument('--out_dir', type=str, default='/path/to/save/train/models', help='Directory for train output.')
    parser.add_argument('--out_dir_test', type=str, default='/path/to/save/inference results', help='Directory for test output.')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='Directory for saved model.')

    ### Input image parameters
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--in_channels', type=int, default=1, help="input channels")
    parser.add_argument('--out_channels', type=int, default=1, help="output channels")
    parser.add_argument('--model_channels', type=int, default=64, help='model channels: default 64')
    parser.add_argument('--num_res_blocks', type=tuple, default=[2, 2, 2, 2], help="Number of residual blocks")
    parser.add_argument('--attention_resolutions', type=tuple, default=(64,32,16,8), help="Attention resolutions")
    parser.add_argument('--cond_lq', type=utils.boolean_string, default=True, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')


    # Arguments for Diffusion
    parser.add_argument('--sf', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--schedule_name', type=str, default='exponential', help='Name of the schedule')
    parser.add_argument('--schedule_kwargs', type=dict, default={'power': 0.3}, help='Additional schedule arguments as a dictionary')
    parser.add_argument('--etas_end', type=float, default=0.99,  help='Ending value for etas')
    parser.add_argument('--steps', type=int, default=4, help='Number of steps')
    parser.add_argument('--min_noise_level', type=float, default=0.2, help='Minimum noise level')
    parser.add_argument('--kappa', type=float, default=2.0, help='Kappa value')
    parser.add_argument('--weighted_mse', action='store_true', help='Use weighted mean squared error')
    parser.add_argument('--predict_type', type=str, default='xstart', help='Type of prediction')
    parser.add_argument('--timestep_respacing', type=int, default=None, help='Timestep respacing value')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--normalize_input', action='store_true',  help='Whether to normalize input')
    parser.add_argument('--latent_flag', action='store_true', help='Flag for using latent variables')


    ### Training and validation parameters
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='size of batches in training')
    parser.add_argument('--test_batch_size', type=int, default=2, help='size of batches in inference')
    parser.add_argument('--betas_G', type=tuple, default=(0.9, 0.999), help='learning rate')
    parser.add_argument('--train_num_workers', type=int, default=16, help='number of CPU to load data')

    # Learning rate configuration
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=2e-5, help='Minimum learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosin', help='Learning rate schedule type')
    parser.add_argument('--warmup_iterations', type=int, default=5000, help='Number of warmup iterations')

    # Dataloader configuration
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--prefetch_factor', type=int, default=8, help='Prefetch factor for data loading')

    # Optimization settings
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='Exponential moving average rate')

    parser.add_argument('--loss_coef', type=float, nargs=2, default=[4.0, 1.0],
                        help='Loss coefficients for [mse, lpips]')
    # Training settings
    parser.add_argument('--use_amp', type=bool, default=False, help='Use Automatic Mixed Precision (AMP) for training')
    parser.add_argument('--seed', type=int, default=123456, help='Random seed for reproducibility')
    parser.add_argument('--global_seeding', type=bool, default=False, help='Use global seeding')


    torch.cuda.empty_cache()
    torch.cuda.synchronize()


    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()
    torch.set_float32_matmul_precision('high')


    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
            torch.cuda.manual_seed_all(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    if FLAGS.train:
        if FLAGS.resume:
            log_file = os.path.join(FLAGS.out_dir, 'log_resume.txt')
            print("Logging to {}\n".format(log_file))
            sys.stdout = utils.StdOut(log_file)
        else:
            utils.clear_folder(FLAGS.out_dir)
            log_file = os.path.join(FLAGS.out_dir, 'log.txt')
            print("Logging to {}\n".format(log_file))
            sys.stdout = utils.StdOut(log_file)
    else:
        utils.clear_folder(FLAGS.out_dir_test)
        log_file = os.path.join(FLAGS.out_dir_test, 'log1.txt')
        print("Logging to {}\n".format(log_file))
        sys.stdout = utils.StdOut(log_file)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch built with): {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # print("Similar to out3 with my perceptual loss function\nSelf Att. Unet\nNN loss and LeakyReLU activations.")
    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)

    main()


