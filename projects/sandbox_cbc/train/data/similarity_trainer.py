import os

import torch
from data import JitteredSignalDataset
from lightning.pytorch import Trainer, callbacks, loggers

from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import ResNet
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler


def main():
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 500
    batches_per_epoch = 20
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
    resnet_context_dim = 100
    resnet_layers = [4, 4]
    resnet_norm_groups = 8
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
    )
    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    #ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-50-transforms-2-2-2-resnet-wider-dl/version_0/checkpoints/"
    #checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #flow_obj.load_state_dict(checkpoint['state_dict'])
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
    print("##### Initialized data loader, calling setup ####")
    jittered_dataset.setup(None)

    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=50, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    outdir = os.getenv("BASE_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="vicreg-training")
    print("##### Initializing trainer #####")
    trainer = Trainer(
        max_epochs=1000,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=10.0,
    )
    trainer.fit(model=embedding, datamodule=jittered_dataset)

if __name__ == '__main__':
    main()
