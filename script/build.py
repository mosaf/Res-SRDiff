import os
import pickle
from time import time
from tqdm import tqdm
import torch
import lpips
import torchvision.utils as vutils
import numpy as np

from networks.unet import UNetModelSwin
import torch.amp as amp
from contextlib import nullcontext
from diffusion.Gaussian_model import create_gaussian_diffusion
import torch.distributed as dist
from copy import deepcopy
from collections import OrderedDict
import h5py


from piq import vif_p, gmsd, psnr, ssim


class Model(object):
    def __init__(self,
                 name,
                 device,
                 data_loader,
                 test_data_loader,
                 FLAGS):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.flags = FLAGS
        self.state = {'epoch': 0, 'step': 0}  # Initialize state
        assert self.name == 'Super_resolution'

        self.setup_dist()

        self.build_model()
        self.create_optim()

        if self.flags.resume:

            print(f"=> Loaded checkpoint from {self.flags.out_dir}")
            checkpoint_file = os.path.join(self.flags.out_dir, 'content.pth')
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.init_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['model_state_dict'])
            torch.cuda.empty_cache()

            # load G
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))

            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = os.path.join(self.flags.out_dir, 'ema_model.pth')
                print((f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}"))
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                self._load_ema_state(self.ema_state, ema_ckpt)
            torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in checkpoint:
                    self.amp_scaler.load_state_dict(checkpoint["amp_scaler"])
                    if self.rank == 0:
                        print("Loading scaler from resumed state...")
        else:
            self.global_step, self.init_epoch = 0, 0


        self.autoencoder = None

        params_diffusion = {
            'sf': self.flags.sf, # default=1.0, Scale factor
            'schedule_name': self.flags.schedule_name, #  default='exponential', Name of the schedule
            'schedule_kwargs': self.flags.schedule_kwargs,  # default={'power': 0.3}, Additional schedule arguments as a dictionary
            'etas_end': self.flags.etas_end,  # default=0.99, Ending value for etas
            'steps': self.flags.steps,  # default=4, Number of steps
            'min_noise_level': self.flags.min_noise_level, # default=0.2, Minimum noise level
            'kappa': self.flags.kappa,  # default=2.0, Kappa value
            'weighted_mse': self.flags.weighted_mse, # action='store_true', Use weighted mean squared error
            'predict_type': self.flags.predict_type,  # default='xstart', Type of prediction
            'timestep_respacing': self.flags.timestep_respacing, # default=None,  Timestep re-spacing value
            'scale_factor': self.flags.scale_factor, # default=1.0, help='Scale factor'
            'normalize_input': self.flags.normalize_input, # action='store_true',  Whether to normalize input
            'latent_flag': self.flags.latent_flag # action='store_true', Flag for using latent variables
        } # parameters of the diffusion model
        self.base_diffusion = create_gaussian_diffusion(**params_diffusion)

        self.lpips_loss = self.creat_lpip_loss()

    def _load_ema_state(self, ema_state, ckpt):
        for key in ema_state.keys():
            if key not in ckpt and key.startswith('module'):
                ema_state[key] = deepcopy(ckpt[7:].detach().data)
            elif key not in ckpt and (not key.startswith('module')):
                ema_state[key] = deepcopy(ckpt['module.' + key].detach().data)
            else:
                ema_state[key] = deepcopy(ckpt[key].detach().data)

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()
        self.num_gpus = num_gpus
        self.rank = 0


    def build_model(self):
        model = UNetModelSwin(image_size = self.flags.image_size,
                              in_channels = self.flags.in_channels,
                              model_channels = self.flags.model_channels,
                              out_channels = self.flags.out_channels,
                              num_res_blocks = self.flags.num_res_blocks,
                              attention_resolutions = self.flags.attention_resolutions,
                              lq_size=self.flags.image_size,
                              )
        model.to(self.device)

        self.model = model

        # EMA
        if self.rank == 0 and self.flags.ema_rate is not None:
            self.ema_model = deepcopy(model).to(self.device)
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data).float() for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]
        else:
            print("EMA is not used, as ema_rate is not defined.")

        # model information
        self.print_model_info()

    def print_model_info(self):
        if self.rank == 0:

            num_params = 0
            for param in self.model.parameters():
                num_params += param.numel()

            num_params = num_params / 1000**2
            print(f"Number of parameters: {num_params:.2f}M")


    def create_optim(self):
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.flags.lr,
                                         betas=self.flags.betas_G, weight_decay=self.flags.weight_decay)
        # amp settings
        self.amp_scaler = amp.GradScaler('cuda') if self.flags.use_amp else None

    def creat_lpip_loss(self):
        lpips_loss = lpips.LPIPS(net='alex', verbose=False).to(self.device)
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        return lpips_loss


    def train(self,
              epochs
              ):
        print(f"Number of total steps {len(self.data_loader) * self.flags.batch_size}")

        total_loss = {}
        total_loss['mse'] = []
        total_loss['lpip'] = []

        total_loss['mse_sample'] = []
        total_loss['lpip_sample'] = []
        total_loss['train_time'] = []
        total_loss['valid_time'] = []

        best_loss = None
        s_time_all = time()
        for epoch in range(self.init_epoch, epochs):
            loss_mse_running, loss_lpip_runnuing = 0., 0.
            self.current_epoch = epoch  + 1
            self.model.train()
            loop = tqdm(self.data_loader, ascii=True, desc=f'Epoch [{epoch + 1}/{epochs}]')
            s_time_epoch = time()
            for _jj, data in enumerate(loop):


                s_time = time()
                hq = data['hq'].to(self.device, dtype=torch.float)
                lq = data['lq'].to(self.device, dtype=torch.float)

                tt = torch.randint(
                    0, self.flags.steps,
                    size=(self.flags.batch_size,),
                    device=self.device,
                )
                if self.flags.cond_lq:
                    model_kwargs = {'lq': lq, }
                else:
                    model_kwargs = None

                noise = torch.randn(
                    size=hq.shape,
                    device=self.device,
                )

                compute_losses = self.base_diffusion.training_losses(
                    model=self.model,
                    x_start=hq,
                    y=lq,
                    t=tt,
                    first_stage_model=self.autoencoder,
                    model_kwargs=model_kwargs,
                    noise=noise,
                )
                if self.flags.num_gpus <= 1:
                    losses, z0_pred, z_t = self.backward_step(compute_losses, hq)
                else:
                    with self.model.no_sync():
                        losses, z0_pred, z_t = self.backward_step(compute_losses, hq)

                if self.flags.use_amp:
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    self.optimizer.step()

                # grad zero
                self.model.zero_grad()
                self.update_ema_model()
                with torch.no_grad():
                    loop.set_postfix(
                                     mse=float(losses['mse'].mean()),
                                     lpips = float(losses['lpips'].mean()),
                                     )
                    loss_mse_running += float(losses['mse'].mean())
                    loss_lpip_runnuing += float(losses['lpips'].mean())

                    total_loss['mse_sample'].append(float(losses['mse'].mean()))
                    total_loss['lpip_sample'].append(float(losses['lpips'].mean()))
                    total_loss['train_time'].append(float(time() - s_time))

            print(
                f"\nMSE: {loss_mse_running / len(self.data_loader):.4f}, "
                f"LPIPs: {loss_lpip_runnuing / len(self.data_loader):.4f}, "
            )

            with torch.no_grad():
                total_loss['mse'].append(loss_mse_running / len(self.data_loader))
                total_loss['lpip'].append(loss_lpip_runnuing / len(self.data_loader))

            print(f"Epoch {epoch + 1} completed in {(time() - s_time_epoch) / 60:.2f} (min)")
            if (epoch + 1) % 5 == 0 or (epoch + 1) in [1, epochs]:
                # validation phase
                self.validation(batch_idx=epoch)

            self.save_to(path=self.flags.out_dir, name=self.flags.model_name)

        print(f"{total_loss['mse'] = }")
        print(f"{total_loss['lpip'] = }")

        with open(os.path.join(self.flags.out_dir, f'metrics_all.pkl'), 'wb') as file_all:
            # Save the dictionary to the file
            pickle.dump(total_loss, file_all)

        total_time = time() - s_time_all
        print('Total train time: {:.2f} (hrs)'.format(total_time / 3600.))

    @torch.no_grad()
    def update_ema_model(self):
        if self.flags.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.flags.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1 - rate)

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate=1):
        loss_coef = self.flags.loss_coef
        context = torch.amp.autocast('cuda') if self.flags.use_amp else nullcontext()
        # diffusion loss
        with context:
            losses, z_t, z0_pred = dif_loss_wrapper
            x0_pred = z0_pred

            # classification loss
            losses["lpips"] = self.lpips_loss(
                x0_pred.clamp(-1.0, 1.0).repeat(1, 3, 1, 1),
                micro_data.repeat(1, 3, 1, 1),
            ).to(x0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)

            losses["mse"] *= loss_coef[0]
            losses["lpips"] *= loss_coef[1]

            assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, x0_pred, z_t

    def validation(self, batch_idx):
        print("Performing validation ...")
        if self.rank == 0:

            self.model.eval()

            indices = np.linspace(
                0,
                self.flags.steps,
                self.flags.steps if self.flags.steps < 5 else 4,
                endpoint=False,
                dtype=np.int64,
            ).tolist()
            if not (self.flags.steps - 1) in indices:
                indices.append(self.flags.steps - 1)

            metrics = {}
            metrics['psnr_before'] = []
            metrics['psnr_after'] = []

            metrics['ssim_before'] = []
            metrics['ssim_after'] = []

            metrics['lpip_before'] = []
            metrics['lpip_after'] = []
            s_time_validation = time()

            data = next(iter(self.test_data_loader))
            im_lq = data['lq'].to(self.device, dtype=torch.float)
            im_gt = data['hq'].to(self.device, dtype=torch.float)

            num_iters = 0
            if self.flags.cond_lq:
                model_kwargs = {'lq': im_lq, }
            else:
                model_kwargs = None
            tt = torch.tensor(
                [self.flags.steps, ] * im_lq.shape[0],
                dtype=torch.int64,
            ).cuda()
            for sample in self.base_diffusion.p_sample_loop_progressive(
                    y=im_lq,
                    model=self.model,
                    # model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=True if self.autoencoder is None else False,
                    model_kwargs=model_kwargs,
                    device=f"cuda:{self.rank}",
                    progress=False,
            ):
                sample_decode = {}
                if num_iters in indices:
                    for key, value in sample.items():
                        if key in ['sample', ]:
                            sample_decode[key] = value.clamp(-1.0, 1.0)
                    im_sr_progress = sample['sample']
                    if num_iters + 1 == 1:
                        im_sr_all = im_sr_progress
                    else:
                        im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                num_iters += 1
                tt -= 1

            viz_sample = torch.cat((im_gt * 0.5 + 0.5,
                                    im_lq * 0.5 + 0.5,
                                    sample_decode['sample'] * 0.5 + 0.5,
                                    ),
                                   dim=0)  # unet
            vutils.save_image(viz_sample,
                              os.path.join(self.flags.out_dir, 'samples_all_{}.png'.format(batch_idx + 1)),
                              nrow=self.test_data_loader.batch_size,
                              normalize=True)

            with h5py.File(os.path.join(self.flags.out_dir, 'samples_all_{}.h5'.format(batch_idx + 1)), 'w') as hf:
                hf.create_dataset('hq', data=im_gt.cpu().numpy())
                hf.create_dataset('lq', data=im_lq.cpu().numpy())

                hf.create_dataset('pred_hq', data=im_sr_all.cpu().numpy())

            print(f"Validation done in {(time() - s_time_validation) / 60:.2f} min")
            # if not (self.flags.use_ema_val and self.flags.ema_rate is not None):
            self.model.train()

    @staticmethod
    def minimax_4d(X, min_value=0, max_value=1.0):
        _max_value = torch.amax(X, dim=(2, 3), keepdim=True)
        _min_value = torch.amin(X, dim=(2, 3), keepdim=True)

        X_std = (X - _min_value) / (_max_value - _min_value)
        return X_std * (max_value - min_value) + min_value

    def inference_(self, batch_size = None, out_dir = None):
        if batch_size is None:
            batch_size = self.flags.test_batch_size

        print(f"Number of total steps {len(self.test_data_loader) * batch_size}")
        if out_dir is None:
            out_dir = self.flags.out_dir_test

        print("Performing validation ...")

        self.model.eval()

        metrics = {}
        metrics['psnr_before'] = []
        metrics['psnr_after'] = []

        metrics['nmse_before'] = []
        metrics['nmse_after'] = []

        metrics['ssim_before'] = []
        metrics['ssim_after'] = []

        metrics['lpip_before'] = []
        metrics['lpip_after'] = []

        metrics['vif_before'] = []
        metrics['vif_after'] = []

        metrics['gmsd_before'] = []
        metrics['gmsd_after'] = []

        metrics['validation_time'] = []

        with torch.no_grad():
            for ii, data in enumerate(tqdm(self.test_data_loader)):
                im_lq = data['lq'].to(self.device, dtype=torch.float)
                im_gt = data['hq'].to(self.device, dtype=torch.float)

                # num_iters = 0
                if self.flags.cond_lq:
                    model_kwargs = {'lq': im_lq, }
                else:
                    model_kwargs = None

                s_time = time()
                results = self.base_diffusion.p_sample_loop(
                    y=im_lq,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    noise_repeat=False,
                    clip_denoised=True if self.autoencoder is None else False,
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=False,
                )  # This has included the decoding for latent space results


                metrics['validation_time'].append(time() - s_time)

                with h5py.File(os.path.join(out_dir, 'samples_all_{}.h5'.format(ii + 1)), 'w') as hf:
                    hf.create_dataset('img_hr', data=self.to_int(im_gt))
                    hf.create_dataset('img_lq', data=self.to_int(im_lq))
                    hf.create_dataset('img_pred', data=self.to_int(results))

                results_ = results.clip(-1, 1) * .5 + .5
                im_lq_ = im_lq * .5 + .5
                im_gt_ = im_gt * .5 + .5

                metrics['psnr_before'].append(psnr(im_lq_, im_gt_, data_range=1.0).item())
                metrics['psnr_after'].append(psnr(results_, im_gt_, data_range=1.0).item())

                metrics['nmse_before'].append(self.compute_nmse(im_lq_, im_gt_).item())
                metrics['nmse_after'].append(self.compute_nmse(results_, im_gt_).item())

                metrics['ssim_before'].append(ssim(im_lq_, im_gt_, data_range=1.0).item())
                metrics['ssim_after'].append(ssim(results_, im_gt_, data_range=1.0).item())




                metrics['lpip_before'].append(self.lpips_loss(im_lq.repeat(1, 3, 1, 1),
                                                              im_gt.repeat(1, 3, 1, 1)).sum().item())
                metrics['lpip_after'].append(self.lpips_loss(results.repeat(1, 3, 1, 1),
                                                             im_gt.repeat(1, 3, 1, 1)).sum().item())

                metrics['vif_before'].append(vif_p(im_lq_, im_gt_, data_range=1.0).item())
                metrics['vif_after'].append(vif_p(results_, im_gt_, data_range=1.0).item())

                metrics['gmsd_before'].append(gmsd(im_lq_, im_gt_).item())
                metrics['gmsd_after'].append(gmsd(results_, im_gt_).item())


                viz_sample = torch.cat((im_gt * 0.5 + 0.5,
                                        im_lq * 0.5 + 0.5,
                                        results * 0.5 + 0.5,
                                        torch.abs(results - im_gt),
                                        ),
                                       dim=0)  # unet
                vutils.save_image(viz_sample,
                                  os.path.join(out_dir, 'samples_all_{}.png'.format(ii + 1)),  # out_dir_outs
                                  nrow=batch_size,
                                  normalize=True)


            print(f"NMSE: {np.mean(metrics['nmse_before']):.4f} --> {np.mean(metrics['nmse_after']):.4f}")
            print(f"PSNR: {np.mean(metrics['psnr_before']):.4f} --> {np.mean(metrics['psnr_after']):.4f}")
            print(f"SSIM: {np.mean(metrics['ssim_before']):.4f} --> {np.mean(metrics['ssim_after']):.4f}")
            print(f"LPIP: {np.mean(metrics['lpip_before']):.4f} --> {np.mean(metrics['lpip_after']):.4f}")
            print(f"GMSD: {np.mean(metrics['gmsd_before']):.4f} --> {np.mean(metrics['gmsd_after']):.4f}")
            print(f"VIfp: {np.mean(metrics['vif_before']):.4f} --> {np.mean(metrics['vif_after']):.4f}")
            print(f"Average evaluation time : {np.mean(metrics['validation_time']):.4f} seconds")


            with open(os.path.join(out_dir, "qmetrics.pkl"), 'wb') as f:
                pickle.dump(metrics, f)

    # Compute NMSE (custom implementation)
    def compute_nmse(self, dist, ref):
        # NMSE = ||ref - dist||^2 / ||ref||^2
        return torch.sum((ref - dist) ** 2) / torch.sum(ref ** 2)
    def to_int(self, x):
        x = x.clip(-1, 1).detach().cpu().numpy() * 0.5 + 0.5
        x = x * 255.0
        return x.round().astype(np.uint8)
    def load_model_inference(self,
                  path='',
                  name=None,
                  verbose=True):
        if name is None:
            name = 'content.pth'
        if verbose:
            print('\nLoading models and checkpoint from {} ...'.format(name))

        print(f"=> Loaded checkpoint from {path}")
        checkpoint_file = os.path.join(path, name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.cuda.empty_cache()



    def save_to(self,
                path='',
                name=None,
                verbose=True):
        content = {'epoch': self.current_epoch,
                   'args': self.flags,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   }

        if name is None:
            name = self.name

        if verbose:
            print('\nSaving models {} ...'.format(name))
        torch.save(self.model.state_dict(), os.path.join(path, '{}.pt'.format(name)))
        if self.amp_scaler is not None:
            content['amp_scaler'] = self.amp_scaler.state_dict()
        torch.save(content, os.path.join(path, 'content.pth'))

        torch.save(self.ema_state, os.path.join(path, 'ema_model.pth'))

