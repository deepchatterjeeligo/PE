import os

import torch
from data import SignalDataSet, VICRegTestingDataset
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.cli import LightningCLI

from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.architectures.embeddings import ResNet
from ml4gw.nn.resnet import ResNet1D
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_component_mass, nonspin_bbh_chirp_mass_q, nonspin_bbh_component_mass_parameter_sampler, nonspin_bbh_chirp_mass_q_parameter_sampler

# def cli_main():
#     cli = LightningCLI(MaskedAutoRegressiveFlow, SignalDataSet,
#                        run=False, subclass_mode_model=False,
#                        subclass_mode_data=True)

# if __name__ == "__main__":
#     cli_main()

class ProxyOpt:
    def __init__(self, opt_class, **optim_kwargs):
        self.opt_class = opt_class
        self.optim_kwargs = optim_kwargs
    def __call__(self, *args, **kwargs):
        kwargs.update(self.optim_kwargs)
        return self.opt_class(*args, **kwargs)

def main():
    #background_path = os.getenv('DATA_DIR') + "/background.h5"
    background_path = os.getenv('DATA_DIR') + "/background-1241143878-20000.hdf5"
    ifos = ["H1", "L1"]
    batch_size = 1000
    batches_per_epoch = 200
    sample_rate = 512
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    highpass = 25
    valid_frac = 0.2
    learning_rate = 7.54e-4
    resnet_context_dim = 8
    resnet_layers = [5, 3, 3]
    resnet_norm_groups = 16
    embedding_dim_multiplier = 3

    inference_params = [
        #"mass_1",
        #"mass_2",
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
       # "ra",
    ]
    num_transforms = 80
    num_blocks = 8
    hidden_features = 150

    optimizer = ProxyOpt(torch.optim.AdamW, weight_decay=0.0152)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
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
    #ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-4-pre-trained-embedding/version_0/checkpoints/epoch=295-step=59200.ckpt"
    #checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #flow_obj.load_state_dict(checkpoint['state_dict'])
    # data
    sig_dat = SignalDataSet(
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
    #sig_dat.setup(None)

    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=50, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=1, monitor="valid_loss", mode="min")
    outdir = os.getenv("SCRATCH_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-80-transforms-8-blocks-hl-upto-3gpc-512Hz", version='533-context-dim-8')
    print("##### Initializing trainer #####")
    trainer = Trainer(
        max_epochs=1000,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor, checkpoint_callback],
        logger=logger,
        gradient_clip_val=10.0,
    )
    #trainer.fit(model=flow_obj, datamodule=sig_dat, ckpt_path=None)
    # Load best model checkpoint
    #best_model_ckpt = torch.load(checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage)
    #flow_obj.load_state_dict(best_model_ckpt['state_dict'])
    
    # Test across the same priors as training
    #trainer.test(model=flow_obj, datamodule=sig_dat)#, ckpt_path='/scratch/bcse/deep1018/pl-logdir/phenomd-80-transforms-6-blocks-hl-upto-2gpc-512Hz/646-pretrained/checkpoints/epoch=227-step=45600.ckpt')

    # Test across all delta function distributions
    print("##### Starting Test with best checkpoint and Delta function priors #####")
    testing_dataset = VICRegTestingDataset(
        os.getenv('DATA_DIR') + "/background-1241143878-20000.hdf5",
        ifos,
        valid_frac,
        1,
        201,
        sample_rate,
        time_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    testing_dataset.setup(None)
    trainer.test(model=flow_obj, datamodule=testing_dataset, ckpt_path='/scratch/bcse/deep1018/pl-logdir/phenomd-80-transforms-8-blocks-hl-upto-3gpc-512Hz/533-context-dim-8/checkpoints/epoch=196-step=39400.ckpt')

if __name__ == '__main__':
    main()
