from typing import Callable, Tuple

import torch
import torch.distributions as dist
from lightning import pytorch as pl
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineAutoregressive
from pyro.nn import ConditionalAutoRegressiveNN

from mlpe.architectures.flows import utils
from mlpe.architectures.flows.flow import NormalizingFlow


class MaskedAutoRegressiveFlow(pl.LightningModule, NormalizingFlow):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        embedding_net: torch.nn.Module,
        opt: torch.optim.SGD,
        sched: torch.optim.lr_scheduler.ConstantLR,
        inference_params: list,
        num_samples_draw: int = 3000,
        num_plot_corner: int = 20,
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_blocks: int = 2,
        activation: Callable = torch.tanh,
    ):
        super().__init__()
        self.param_dim, self.n_ifos, self.strain_dim = shape
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation
        self.optimizer = opt
        self.scheduler = sched
        self.inference_params = inference_params
        self.num_samples_draw = num_samples_draw
        self.num_plot_corner = num_plot_corner
        # define embedding net and base distribution
        self.embedding_net = embedding_net
        # build the transform - sets the transforms attrib
        self.build_flow()

    def transform_block(self):
        """Returns single autoregressive transform"""
        arn = ConditionalAutoRegressiveNN(
            self.param_dim,
            self.context_dim,
            self.num_blocks * [self.hidden_features],
            nonlinearity=self.activation,
        )
        transform = ConditionalAffineAutoregressive(arn)
        return transform

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            torch.zeros(self.param_dim, device=self.device),
            torch.ones(self.param_dim, device=self.device),
        )

    def build_flow(self):
        """Build the transform"""
        self.transforms = []
        for idx in range(self.num_transforms):
            _transform = self.transform_block()
            self.transforms.extend([_transform])
        self.transforms = ConditionalComposeTransformModule(self.transforms)

    def training_step(self, batch, batch_idx):
        strain, parameters, _ = batch
        loss = -self.log_prob(parameters, context=strain).mean()
        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, sync_dist=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        strain, parameters, _ = batch
        loss = -self.log_prob(parameters, context=strain).mean()
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def on_test_epoch_start(self):
        self.test_results = []
        self.num_plotted = 0

    def test_step(self, batch, batch_idx):
        strain, parameters, _ = batch
        res = utils.draw_samples_from_model(
            strain,
            parameters,
            self,
            self.inference_params,
            self.num_samples_draw,
        )
        self.test_results.append(res)
        if batch_idx % 10 == 0 and self.num_plotted < self.num_plot_corner:
            skymap_filename = f"{self.num_plotted}_mollview.png"
            res.plot_corner(
                save=True,
                filename=f"{self.num_plotted}_corner.png",
                levels=(0.5, 0.9),
            )
            utils.plot_mollview(
                res.posterior["phi"] - torch.pi,  # between -pi to pi in healpy
                res.posterior["dec"],
                truth=(
                    res.injection_parameters["phi"] - torch.pi,
                    res.injection_parameters["dec"],
                ),
                outpath=skymap_filename,
            )
            self.num_plotted += 1
            self.print("Made corner plots and skymap for ", batch_idx)

    def on_test_epoch_end(self):
        import bilby

        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename="pp-plot.png",
            keys=self.inference_params,
        )
        del self.test_results, self.num_plotted

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr=torch.cuda.device_count() * 1e-3)
        sched = self.scheduler(opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "valid_loss"}}
