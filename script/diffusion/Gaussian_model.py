

from diffusion import gaussian_diffusion as gd


from diffusion.respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(
    *,
    normalize_input,
    schedule_name,
    sf=4,
    min_noise_level=0.01,
    steps=1000,
    kappa=1,
    etas_end=0.99,
    schedule_kwargs=None,
    weighted_mse=False,
    predict_type='xstart',
    timestep_respacing=None,
    scale_factor=None,
    latent_flag=True,
):
    sqrt_etas = gd.get_named_eta_schedule(
            schedule_name,
            num_diffusion_timesteps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            kwargs=schedule_kwargs,
            )
    if timestep_respacing is None:
        timestep_respacing = steps
    else:
        assert isinstance(timestep_respacing, int)
    if predict_type == 'xstart':
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_type == 'epsilon':
        model_mean_type = gd.ModelMeanType.EPSILON
    elif predict_type == 'epsilon_scale':
        model_mean_type = gd.ModelMeanType.EPSILON_SCALE
    elif predict_type == 'residual':
        model_mean_type = gd.ModelMeanType.RESIDUAL
    else:
        raise ValueError(f'Unknown Predicted type: {predict_type}')
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=gd.LossType.WEIGHTED_MSE if weighted_mse else gd.LossType.MSE,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        sf=sf,
        latent_flag=latent_flag,
    )

