from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from nflows.transforms.normalization import BatchNorm

from mlpe.architectures.flows.flow import NormalizingFlow


@dataclass
class MaskedAutoRegressiveFlow(NormalizingFlow):
    shape: Tuple[int, int, int]
    embedding_net: torch.nn.Module
    num_transforms: int
    hidden_features: int = 50
    num_blocks: int = 2
    activation: Callable = torch.tanh
    use_batch_norm: bool = True
    use_residual_blocks: bool = True
    batch_norm_between_layers: bool = True

    def __post_init__(self):
        self.param_dim, self.n_ifos, self.strain_dim = self.shape
        super().__init__(
            self.param_dim,
            self.n_ifos,
            self.strain_dim,
            embedding_net=self.embedding_net,
            num_flow_steps=self.num_transforms,
        )

    def transform_block(self):
        """Returns the single block of the MAF"""
        single_block = [
            RandomPermutation(features=self.param_dim),
            MaskedAffineAutoregressiveTransform(
                features=self.param_dim,
                hidden_features=self.hidden_features,
                context_features=self.context_dim,
                num_blocks=self.num_blocks,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                use_residual_blocks=self.use_residual_blocks,
            ),
        ]
        if self.batch_norm_between_layers:
            single_block.append(BatchNorm(features=self.param_dim))
        return single_block

    def distribution(self):
        """Returns the base distribution for the flow"""
        return StandardNormal([self.param_dim])

    def build_flow(self):
        transforms = []
        for _ in range(self.num_transforms):
            transforms.extend(self.transform_block())

        transform = CompositeTransform(transforms)
        base_dist = self.distribution()
        self._flow = Flow(transform, base_dist, self.embedding_net)
