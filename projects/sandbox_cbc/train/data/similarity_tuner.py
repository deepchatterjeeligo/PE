import os

import torch
from data import JitteredSignalDataset
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
from ray.train.torch import TorchConfig, TorchTrainer


from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import ResNet
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_cos_theta_parameter_sampler


def train_func(config):
    # background_path = os.getenv('DATA_DIR') + "/background-1241123878-20000.hdf5"
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = config['batch_size']
    batches_per_epoch = 500
    sample_rate = 2048
    time_duration = 4
    f_max = 300
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
    resnet_context_dim = config['resnet_context_dim']
    resnet_layers = [config['layer_1_size'], config['layer_2_size'], config['layer_3_size']]
    inference_params = [
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "cos_dec",
        "psi",
        "phi",
    ]
    _, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=config['resnet_norm_groups'],
        kernel_size=config['kernel_size'],
        learning_rate=config['lr'],
        wt_repr=config['wt_repr'],
        wt_std=config['wt_std'],
        wt_cov=config['wt_cov'],
        embedding_dim_multiplier=config["embedding_dim_multiplier"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    prior_func = nonspin_bbh_chirp_mass_q_cos_theta_parameter_sampler

    # data
    jittered_dataset = JitteredSignalDataset(
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
    jittered_dataset.setup(None)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model=embedding, datamodule=jittered_dataset)


def tune_with_asha(ray_trainer, num_samples=10, num_epochs=15):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=3, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    tuner = tune.Tuner.restore('/scratch/bcse/deep1018/ray_results/time_and_phase_marg_expt/', ray_trainer, resume_unfinished=True, resume_errored=False, restart_errored=False)
    return tuner.fit()

if __name__ == '__main__':
    import torch
    print("CUDA available", torch.cuda.is_available())
    ray.init(configure_logging=False)
    default_config = {
        "layer_1_size": 4,
        "layer_2_size": 4,
        "layer_3_size": 4,
        "resnet_context_dim": 10,
        "resnet_norm_groups": 16,
        "lr": 1e-3,
        "batch_size": 800
    }
    search_space = {
        "batch_size": tune.choice([100, 200, 300, 400]),
        "layer_1_size": tune.choice([3, 5, 7]),
        "layer_2_size": tune.choice([3, 5, 7]),
        "resnet_norm_groups": tune.choice([4, 8, 16]),
        "layer_3_size": tune.choice([3, 5, 7]),
        "kernel_size": tune.choice([3, 5, 7, 9]),
        "wt_repr": tune.choice([1,]),
        "wt_cov": tune.choice([1, 5]),
        "wt_std": tune.choice([1, 5]),
        "resnet_context_dim": tune.choice([8, 9, 10, 11, 12]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "momentum": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "embedding_dim_multiplier": tune.choice([2, 3, 4])
    }
    # The maximum training epochs
    num_epochs = 15

    # Number of samples from parameter space
    num_samples = 1000

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path=os.getenv("SCRATCH_DIR") + "/ray_results",
        name="time_and_phase_marg_expt",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="valid_loss_epoch",
            checkpoint_score_order="min",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
        torch_config=TorchConfig(backend="gloo")
    )
    results = tune_with_asha(ray_trainer, num_samples=num_samples, num_epochs=num_epochs)
    print("Best hyperparameters found were: ", results.get_best_result().config)
