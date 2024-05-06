import os

import torch
from data import JitteredSignalDataset, VICRegTestingDataset
from lightning.pytorch import Trainer, callbacks, loggers

from ml4gw import distributions
from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import ResNet
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler


def nonspin_bbh_chirp_mass_q_parameter_sampler_close(device='cpu'):
    p = nonspin_bbh_chirp_mass_q_parameter_sampler(device=device)
    p.parameters['luminosity_distance'] = distributions.PowerLaw(
        torch.as_tensor(1, device=device, dtype=torch.float32),
        torch.as_tensor(1000, device=device, dtype=torch.float32),
        index=2,
        name="luminosity_distance"
    )
    return p


def main():
    background_path = os.getenv('DATA_DIR') + "/background-1241123878-20000.hdf5"
    ifos = ["H1", "L1"]
    batch_size = 1200
    batches_per_epoch = 500
    sample_rate = 512
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
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
    _, n_ifos, strain_dim = (
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
        learning_rate=7e-4,
        wt_cov=1,
        wt_repr=1,
        wt_std=5,
        embedding_dim_multiplier=embedding_dim_multiplier,
        momentum=8e-5,
        weight_decay=4e-4,
    )
    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    #ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-50-transforms-2-2-2-resnet-wider-dl/version_0/checkpoints/"
    #checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #flow_obj.load_state_dict(checkpoint['state_dict'])
    # data
    jittered_dataset = JitteredSignalDataset(
        os.getenv('DATA_DIR') + "/background-1241143878-20000.hdf5",
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
    jittered_dataset.setup(None)
    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("medium")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=10, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=1, monitor="valid_loss", mode="min")
    outdir = os.getenv("SCRATCH_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="vicreg-training-512-Hz", version='533-context-dim-8')
    print("##### Initializing trainer #####")
    trainer = Trainer(
        max_epochs=50,
        accumulate_grad_batches=3,
        callbacks=[lr_monitor, early_stop_cb, checkpoint_callback],
        logger=logger,
        gradient_clip_val=100.0,
    )
    # ckpt_path = '/scratch/bcse/deep1018/pl-logdir/vicreg-training-512-Hz/344-context-dim-8/checkpoints/epoch=1-step=334.ckpt'
    trainer.fit(model=embedding, datamodule=jittered_dataset, ckpt_path=None)

    del jittered_dataset

    best_model_ckpt = torch.load(checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage)
    embedding.load_state_dict(best_model_ckpt['state_dict'])
    print("##### Starting Test with best checkpoint #####")
    testing_dataset = VICRegTestingDataset(
        background_path,
        ifos,
        valid_frac,
        500,
        3,
        sample_rate,
        time_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    testing_dataset.setup(None)
    trainer.test(model=embedding, datamodule=testing_dataset, ckpt_path=None)

if __name__ == '__main__':
    main()
