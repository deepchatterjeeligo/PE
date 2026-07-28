from typing import Callable
import math
import numpy as np

import torch
from ml4gw.constants import MSUN
from ml4gw.waveforms.cbc import utils
from ml4gw.waveforms.generator import (
    TimeDomainCBCWaveformGenerator,
    EXTRA_CYCLES,
    EXTRA_TIME_FRACTION,
)

from .generator import WaveformGenerator
from ..loader import FrequencyDomainWaveformLoader
from ..sampler import WaveformSampler


class CBCGenerator(WaveformGenerator):
    def __init__(
        self,
        *args,
        approximant: Callable,
        f_min: float,
        f_ref: float,
        right_pad: float,
        **kwargs,
    ):
        """
        A lightweight wrapper around
        `ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator`
        to make it compatible with
        `amplfi.train.data.waveforms.generator.WaveformGenerator`.


        Args:
            *args:
                Positional arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
            approximant:
                A callable that takes parameter tensors
                and returns the cross and plus polarizations.
                For example, `ml4gw.waveforms.IMRPhenomD()`
            f_min:
                Lowest frequency at which waveform signal content
                is generated
            f_ref:
                Reference frequency
            right_pad:
                Position in seconds where coalesence is placed
                relative to the right edge of the window
            **kwargs:
                Keyword arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
        """
        super().__init__(*args, **kwargs)
        self.right_pad = right_pad
        self.approximant = approximant
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant,
            self.sample_rate,
            self.duration,
            f_min,
            f_ref,
            right_pad + self.fduration / 2,
        )

    def forward(self, **parameters) -> torch.Tensor:
        hc, hp = self.waveform_generator(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        hc, hp = waveforms.transpose(1, 0)

        return hc.float(), hp.float()


class TimeDomainCBCWaveformGeneratorFromLoader(TimeDomainCBCWaveformGenerator):
    """Derive the `TimeDomainCBCWaveformGenerator` to allow for loading
    frequency domain waveforms from disk and conditioning them to time
    domain waveforms.

    Args:
        sample_rate:
            Rate at which returned time domain waveform will be
            sampled in Hz. This also specifies ``f_max`` for generating
            waveforms via the nyquist frequency: ``f_max = sample_rate // 2``.
        f_min:
            Lower frequency bound for waveforms
        duration:
            Length of waveform in seconds.
            Waveforms will be left padded with zeros
            appropiately to fill the requested duration
        right_pad:
            How far from the right edge of the window
            in seconds the returned waveform coalescence
            will be placed.
        f_ref:
            Reference frequency for the waveform
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        sample_rate: float,
        duration: float,
        f_min: float,
        f_ref: float,
        right_pad: float,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.frequencies = frequencies
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.duration = duration
        self.right_pad = right_pad
        self.f_ref = f_ref

        self.highpass = self.build_highpass_filter()

    def get_frequencies(self, df: float):
        """Get the frequencies from 0 to nyquist for corresponding df"""
        return torch.from_numpy(self.frequencies).to(
            torch.float32
        )

    def generate_conditioned_fd_waveform(self, pols, parameters):
        """
        Overrides the base class method to sample polarizations from
        file containing frequency-domain waveforms.

        Args:
            N:
                number of waveforms polarizations to sample from file
        """  # noqa: E501
        # convert masses to kg, make sure
        # they are doubles so there is no
        # overflow in the calculations

        mass_1, mass_2 = (
            parameters["mass_1"].double() * MSUN,
            parameters["mass_2"].double() * MSUN,
        )
        total_mass = mass_1 + mass_2
        s1z, s2z = parameters["s1z"], parameters["s2z"]
        device = mass_1.device

        f_isco = utils.frequency_isco(mass_1, mass_2)
        f_min = torch.minimum(
            f_isco,
            torch.tensor(self.f_min, device=device),
        )

        # upper bound on chirp time
        tchirp = utils.chirp_time_bound(f_min, mass_1, mass_2, s1z, s2z)

        # upper bound on final black hole spin
        s = utils.final_black_hole_spin_bound(s1z, s2z)

        # upper bound on the final plunge, merger, and ringdown time
        tmerge = utils.merge_time_bound(
            mass_1, mass_2
        ) + utils.ringdown_time_bound(total_mass, s)

        # extra time to include for all waveforms to take care of situations
        # where the frequency is close to merger (and is sweeping rapidly):
        # this is a few cycles at the low frequency
        textra = EXTRA_CYCLES / f_min

        # lower bound on chirpt frequency start used for
        # conditioning the frequency domain waveform
        fstart = utils.chirp_start_frequency_bound(
            (1.0 + EXTRA_TIME_FRACTION) * tchirp, mass_1, mass_2
        )

        # revised chirp time estimate based on fstart
        tchirp_fstart = utils.chirp_time_bound(
            fstart, mass_1, mass_2, s1z, s2z
        )

        # chirp length in samples
        chirplen = torch.round(
            (tchirp_fstart + tmerge + 2.0 * textra) * self.sample_rate
        )

        # pad to next power of 2
        chirplen = 2 ** torch.ceil(torch.log(chirplen) / math.log(2))

        # get smallest df corresponding to longest chirp length,
        # which will make sure there is no wrap around effects.
        df = 1.0 / (chirplen.max() / self.sample_rate)

        # generate frequency array from 0 to nyquist based on df
        frequencies = self.get_frequencies(df).to(mass_1.device)

        # downselect to frequencies above fstart,
        # and generate the waveform at the specified frequencies
        freq_mask = frequencies >= fstart.min()

        # generate the waveform at specified frequencies
        cross, plus = pols["cross"], pols["plus"]
        batch_size = cross.size(0)

        # create tensors to hold the full spectrum
        # of frequencies from 0 to nyquist, and then
        # fill in the requested frequencies with the waveform values
        shape = (batch_size, frequencies.size(0))
        hc_spectrum = torch.zeros(shape, dtype=cross.dtype, device=device)
        hp_spectrum = torch.zeros(shape, dtype=plus.dtype, device=device)

        hc_spectrum[:, freq_mask] = cross
        hp_spectrum[:, freq_mask] = plus

        # build a taper that is dependent on each
        # individual waveforms fstart;
        # since this means that the taper sizes
        # will be different for each waveform,
        # construct the tapers based on the maximum size
        # and then set the values outside of the individual
        # waveform taper regions to 1.0
        k0s = torch.round(fstart / df)
        k1s = torch.round(f_min / df)

        num_freqs = frequencies.size(0)
        frequency_indices = torch.arange(num_freqs, device=device)
        taper_mask = frequency_indices <= k1s[:, None]
        taper_mask &= frequency_indices >= k0s[:, None]

        indices = frequency_indices.expand(batch_size, -1)

        kvals = indices[taper_mask]
        k0s_expanded = k0s.unsqueeze(1).expand(-1, num_freqs)[taper_mask]
        k1s_expanded = k1s.unsqueeze(1).expand(-1, num_freqs)[taper_mask]

        windows = 0.5 - 0.5 * torch.cos(
            torch.pi * (kvals - k0s_expanded) / (k1s_expanded - k0s_expanded)
        )

        hc_spectrum[taper_mask] *= windows
        hp_spectrum[taper_mask] *= windows

        # zero out frequencies below fstart
        zero_mask = frequencies < fstart[:, None]
        hc_spectrum[zero_mask] = 0
        hp_spectrum[zero_mask] = 0

        # set nyquist frequency to zero
        hc_spectrum[..., -1], hp_spectrum[..., -1] = 0.0, 0.0

        # apply time translation in (i.e. phase shift in frequency domain)
        # that will translate the coalescense time such that it is
        # ``right_pad`` seconds from the right edge of the window
        tshift = round(self.right_pad * self.sample_rate) / self.sample_rate
        kvals = torch.arange(num_freqs, device=device)
        phase_shift = torch.exp(1j * 2 * torch.pi * df * tshift * kvals)

        hc_spectrum *= phase_shift
        hp_spectrum *= phase_shift

        return hc_spectrum, hp_spectrum, parameters

    def apply_td_condition_stage2(
        self,
        hc: torch.Tensor,
        hp: torch.Tensor,
        parameters: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply stage 2 TD conditioning following XLALSimInspiralTDConditionStage2.

        Tapers the first 1/(f_min * dt) samples and the
        last 1/(f_isco * dt) samples with a cosine window.

        See https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/python/lalsimulation/gwsignal/core/conditioning_subroutines.py#L90
        """  # noqa: E501
        mass_1 = parameters["mass_1"].double() * MSUN
        mass_2 = parameters["mass_2"].double() * MSUN
        f_isco = utils.frequency_isco(mass_1, mass_2)

        N = hp.shape[-1]
        min_taper_samples = 4

        # Taper start of signal
        ntaper_start = max(
            round(self.sample_rate / self.f_min), min_taper_samples
        )
        t = torch.arange(ntaper_start).type_as(hp)
        w_start = 0.5 - 0.5 * torch.cos(torch.pi * t / ntaper_start)
        hp[:, :ntaper_start] *= w_start
        hc[:, :ntaper_start] *= w_start

        # Taper end of signal, vectorized because f_isco is different
        # for each waveform in the batch
        ntaper_end = torch.clamp(
            torch.round(self.sample_rate / f_isco), min=min_taper_samples
        )
        max_ntaper = int(ntaper_end.max().item())

        idx = (max_ntaper - 1) - torch.arange(max_ntaper - 1).type_as(hp)
        cos_arg = torch.pi * idx[None, :] / ntaper_end[:, None]
        w_end = 0.5 - 0.5 * torch.cos(cos_arg)
        in_taper = idx[None, :] < ntaper_end[:, None]
        w_end = torch.where(in_taper, w_end, torch.ones_like(w_end))

        hp[:, N - max_ntaper + 1 :] *= w_end
        hc[:, N - max_ntaper + 1 :] *= w_end

        return hc, hp

    def forward(
        self,
        pols: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ):
        """
        Generates a time-domain waveform from a frequency-domain approximant.
        Conditioning is based onhttps://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/python/lalsimulation/gwsignal/core/waveform_conditioning.py?ref_type=heads#L248

        A frequency domain waveform is generated, conditioned
        (see ``generate_conditioned_fd_waveform``) and fft'd into the time-domain

        Args:
            pols:
                Dictionary containing the cross and plus polarizations
            parameters:
                Dictionary containing the waveform parameters
        """  # noqa: E501

        hc, hp, parameters = self.generate_conditioned_fd_waveform(pols, parameters)

        # fft to time domain and apply appropriate scaling
        hc = torch.fft.irfft(hc) * self.sample_rate
        hp = torch.fft.irfft(hp) * self.sample_rate

        hc, hp = self.apply_td_condition_stage2(hc, hp, parameters)

        # pad waveforms on left up to requested duration
        pad = int((self.duration * self.sample_rate) - hp.shape[-1])
        hc = torch.nn.functional.pad(hc, (pad, 0))
        hp = torch.nn.functional.pad(hp, (pad, 0))

        # finally, highpass the waveforms,
        # going to double precision
        hp = self.highpass(hp.double())
        hc = self.highpass(hc.double())

        return hc, hp, parameters


class CBCGeneratorFromLoader(FrequencyDomainWaveformLoader):
    def __init__(
        self,
        *args,
        f_min: float,
        f_ref: float,
        right_pad: float,
        **kwargs,
    ):
        """
        A lightweight wrapper around
        `ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator`
        to make it compatible with
        `amplfi.train.data.waveforms.generator.WaveformGenerator`.


        Args:
            *args:
                Positional arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
            f_min:
                Lowest frequency at which waveform signal content
                is generated
            f_ref:
                Reference frequency
            right_pad:
                Position in seconds where coalesence is placed
                relative to the right edge of the window
            **kwargs:
                Keyword arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
        """
        super().__init__(*args, **kwargs)
        self.right_pad = right_pad
        self.waveform_generator = TimeDomainCBCWaveformGeneratorFromLoader(
            self.frequencies,
            self.sample_rate,
            self.duration,
            f_min,
            f_ref,
            right_pad + self.fduration / 2,
        )

    def forward(self, N) -> torch.Tensor:
        pols, parameters = self.sample(N)
        hc, hp, parameters = self.waveform_generator(pols, parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        hc, hp = waveforms.transpose(1, 0)

        return hc.float(), hp.float(), parameters
