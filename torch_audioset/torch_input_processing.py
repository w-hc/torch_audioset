import numpy as np
import torch
import torchaudio.transforms as ta_trans
from .params import CommonParams


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

    def __call__(self, waveform, sample_rate):
        '''
        Args:
            waveform: torch tsr [1, num_channels, num_steps]
            sample_rate: per second sample rate
        '''
        print("please confirm whether wave is torch or numpy tsr: currently {}".format(type(waveform)))
        waveform = waveform.squeeze(axis=0)

        # init the ops
        resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)

        x = waveform.mean(axis=0, keepdims=True)  # average over channels
        x = resampler(x)
        x = self.mel_trans_ope(x)  # [1, num_freq, time_steps]
        x = x.squeeze(dim=0).T  # [time_steps, num_freq]

        window_size_in_frames = int(round(
            CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
        ))

        # reshape into chunks of non-overlapping sliding window
        length = len(x)
        num_chunks = length // window_size_in_frames
        num_frames_to_use = num_chunks * window_size_in_frames
        x = x[:num_frames_to_use, :]
        x = x.reshape(num_chunks, window_size_in_frames, x.shape[-1])
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
        specgram = specgram ** 0.5  # TODO: document this hack later!!
        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + CommonParams.LOG_OFFSET)
        return mel_specgram
