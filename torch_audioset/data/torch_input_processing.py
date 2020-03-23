import numpy as np
import torch
import torchaudio.transforms as ta_trans
from ..params import CommonParams


class WaveformToInput():
    def __init__(self):
        audio_sample_rate = CommonParams.TARGET_SAMPLE_RATE
        window_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_WINDOW_LENGTH_SECONDS
        ))
        hop_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_HOP_LENGTH_SECONDS
        ))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        assert window_length_samples == 400
        assert hop_length_samples == 160
        assert fft_length == 512
        self.mel_trans_ope = VGGishLogMelSpectrogram(
            CommonParams.TARGET_SAMPLE_RATE, n_fft=fft_length,
            win_length=window_length_samples, hop_length=hop_length_samples,
            f_min=CommonParams.MEL_MIN_HZ,
            f_max=CommonParams.MEL_MAX_HZ,
            n_mels=CommonParams.NUM_MEL_BANDS
        )
        # note that the STFT filtering logic is exactly the same as that of a
        # conv kernel. It is the center of the kernel, not the left edge of the
        # kernel that is aligned at the start of the signal.

    def __call__(self, waveform, sample_rate):
        '''
        Args:
            waveform: torch tsr [num_audio_channels, num_time_steps]
            sample_rate: per second sample rate
        Returns:
            batched torch tsr of shape [N, C, T]
        '''
        waveform = torch.from_numpy(waveform)
        # TODO move the waveform data to GPU and see if there is a speed boost

        x = waveform.mean(axis=0, keepdims=True)  # average over channels
        resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)
        x = resampler(x)
        x = self.mel_trans_ope(x)
        x = x.squeeze(dim=0)  # # [1, C, T] -> [C, T]

        window_size_in_frames = int(round(
            CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
        ))
        num_chunks = x.shape[1] // window_size_in_frames

        # reshape into chunks of non-overlapping sliding window
        num_frames_to_use = num_chunks * window_size_in_frames
        x = x[:, :num_frames_to_use]
        x = x.reshape(-1, num_chunks, window_size_in_frames)
        x = x.permute(1, 0, 2)  # [C, N, T] ->  [N, C, T]
        return x


class VGGishLogMelSpectrogram(ta_trans.MelSpectrogram):
    '''
    This is a _log_ mel-spectrogram transform that adheres to the transform
    used by Google's vggish model input processing pipeline
    '''
    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time)
        """
        specgram = self.spectrogram(waveform)
        # NOTE at mel_features.py:98, googlers used np.abs on fft output and
        # as a result, the output is just the norm of spectrogram raised to power 1
        # For torchaudio.MelSpectrogram, however, the default
        # power for its spectrogram is 2.0. Hence we need to sqrt it.
        # I can change the power arg at the constructor level, but I don't
        # want to make the code too dirty
        specgram = specgram ** 0.5

        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + CommonParams.LOG_OFFSET)
        return mel_specgram
