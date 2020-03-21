import numpy as np
import torch
import torchaudio as ta
from torch_audioset.torch_input_processing import WaveformToInput as TorchTransform
from torch_audioset.tflow_input_processing import WaveformToInput as TflowTransform


def sinusoidal_data():
    # Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
    # to test resampling to 16 kHz during feature extraction).
    num_secs = 30
    freq = 1000
    sample_rate = 44100
    t = np.linspace(0, num_secs, int(num_secs * sample_rate))
    x = np.sin(2 * np.pi * freq * t)
    expected_stats = {
        'embedding_mean': 0.131,
        'embedding_std': 0.238,
        'post_pca_mean': 123.0,
        'post_pca_std': 75.0
    }
    # float32, [C=2, T=3*sr]
    x = x.astype(np.float32).reshape(1, -1).repeat(axis=0, repeats=2)
    return x, sample_rate, expected_stats


def real_siren_data():
    x, sr = ta.load('/home-nfs/whc/siren.wav')
    x = x.numpy()
    return x, sr


def main():
    # x, sr, _ = sinusoidal_data()
    x, sr = real_siren_data()
    torch_data = TorchTransform()(x, sr)
    tflow_data = TflowTransform(method='vggish')(x, sr)
    assert torch_data.shape == tflow_data.shape
    print("mean torch {:.5f} vs tflow {:.5f}".format(
        torch_data.mean(), tflow_data.mean()
    ))
    print("std  torch {:.5f} vs tflow {:.5f}".format(
        torch_data.std(), tflow_data.std()
    ))


if __name__ == "__main__":
    main()


'''
Input processing: torch vs vggish tflow
1. on sinusoidal test data, the two differ by a little bit.
    sqrting the STFT magnitude matters little
    When sqrt is on:
        mean torch -3.42271 vs tflow -3.26085
        std  torch 2.12663 vs tflow 2.05112
    When sqrt is off:
        mean torch -3.63301 vs tflow -3.26085
        std  torch 2.95379 vs tflow 2.05112
2. However, on real siren data, sqrting the STFT is critical
    When sqrt is on:
        mean torch -0.87242 vs tflow -0.85927
        std  torch 1.37589 vs tflow 1.37079
    When sqrt is off:
        mean torch -2.13134 vs tflow -0.85927
        std  torch 2.46712 vs tflow 1.37079

'Tis clear that you have to sqrt the STFT magnitude
'''
