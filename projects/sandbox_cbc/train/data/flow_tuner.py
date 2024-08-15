import os

import torch
from data import SignalDataSet
from lightning.pytorch import Trainer

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer


from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler


class ProxyOpt:
    def __init__(self, opt_class, **optim_kwargs):
        self.opt_class = opt_class
        self.optim_kwargs = optim_kwargs
    def __call__(self, *args, **kwargs):
        kwargs.update(self.optim_kwargs)
        return self.opt_class(*args, **kwargs)


def train_func(config):
    background_path = os.getenv('DATA_DIR') + "/background-1241123878-40000.h5"
    ifos = ["H1", "L1"]
    batch_size = config['batch_size']
    batches_per_epoch = 100
    sample_rate = 512
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
    learning_rate = config['learning_rate']
    resnet_context_dim = 8
    resnet_layers = [5, 3, 3]
    resnet_norm_groups = 16
    embedding_dim_multiplier = 3
    inference_params = [
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
    ]
    num_transforms = config['num_transforms']
    num_blocks = config['num_blocks']
    hidden_features = config['hidden_features']

    optimizer_kwargs = dict(weight_decay=config['weight_decay'])
    if 'SGD' in config['optimizer_class'].__name__:
        optimizer_kwargs.update({'momentum': config['momentum']})

    optimizer = ProxyOpt(config['optimizer_class'], **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
        kernel_size=5,
        embedding_dim_multiplier=embedding_dim_multiplier,
    )
    embedding_ckpt_path = os.getenv("SCRATCH_DIR") + "/pl-logdir/vicreg-training-512-Hz/533-context-dim-8/checkpoints/epoch=13-step=2338.ckpt"
    embedding_ckpt = torch.load(embedding_ckpt_path, map_location=lambda storage, loc: storage)
    embedding.load_state_dict(embedding_ckpt['state_dict'])

    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    flow_obj = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        num_transforms=num_transforms,
        num_blocks=num_blocks,
        hidden_features=hidden_features,
        learning_rate=learning_rate
    )

    signal_dataset = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        sample_rate,
        time_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    print("##### Initialized data loader, calling setup ####")
    signal_dataset.setup(None)
    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model=flow_obj, datamodule=signal_dataset)


def tune_with_asha(ray_trainer, scheduler, num_samples=10):
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="avg_valid_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    #tuner = tune.Tuner.restore('/u/deep1018/ray_results/flow_tune_optimizer_expt/', trainable=ray_trainer)
    return tuner.fit()

if __name__ == '__main__':
    import torch
    import socket
    print("CUDA present {} on {}".format(torch.cuda.is_available(), socket.gethostname()))
    ray.init(configure_logging=False)

    search_space = {
        "num_transforms": tune.choice([60, 80, 100]),
        "num_blocks": tune.choice([6, 7, 8]),
        "hidden_features": tune.choice([100, 120, 150]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([800, 1000, 1200]),
        "momentum": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "optimizer_class": tune.choice([torch.optim.AdamW, torch.optim.SGD])
    }
    # The maximum training epochs
    num_epochs = 30

    # Number of samples from parameter space
    num_samples = 100

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=5, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path=os.getenv("SCRATCH_DIR") + "/ray_results",
        name="flow_tune_optimizer_expt",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="avg_valid_loss",
            checkpoint_score_order="min",
        ),
    )
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    results = tune_with_asha(ray_trainer, scheduler, num_samples=num_samples)
    print("Best hyperparameters found were: ", results.get_best_result().config)
