# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from torch_audioset.data.torch_input_processing import WaveformToInput as TorchTransform
from torch_audioset.params import YAMNetParams
from torch_audioset.yamnet.model import yamnet as torch_yamnet
from torch_audioset.yamnet.model import yamnet_category_metadata


def sf_load_from_int16(fname):
    x, sr = sf.read(fname, dtype='int16', always_2d=True)
    x = x / 2 ** 15
    x = x.T.astype(np.float32)
    return x, sr


if __name__ == '__main__':
    # one wav file as argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('wav')
    args = parser.parse_args()

    wav_fname = args.wav
    waveforms, sr = sf_load_from_int16(wav_fname)
    waveform = waveforms[0]
    waveform_for_torch = torch.tensor(waveforms)
    patches, spectrogram = TorchTransform().wavform_to_log_mel(waveform_for_torch, 16000)

    pt_model = torch_yamnet(pretrained=False)
    # Manually download the `yamnet.pth` file.
    pt_model.load_state_dict(torch.load('./yamnet.pth'))

    with torch.no_grad():
        pt_model.eval()
        # x = torch.from_numpy(patches)
        # x = x.unsqueeze(1)  # [5, 96, 64] -> [5, 1, 96, 64]
        x = patches
        pt_pred = pt_model(x, to_prob=True)
        pt_pred = pt_pred.numpy()

    scores = pt_pred
    params = YAMNetParams()
    class_names = [x['name'] for x in yamnet_category_metadata()]

    # Visualize the results.
    plt.figure(figsize=(10, 8))

    # Plot the waveform.
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    # Plot the log-mel spectrogram (returned by the model).
    plt.subplot(3, 1, 2)
    extent = (0, spectrogram.shape[0], -0.5, spectrogram.shape[1] - 0.5)
    plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='lower', extent=extent)
    plt.xlim([0, len(waveform) / sr / YAMNetParams.STFT_HOP_SECONDS])

    # Plot and label the model output scores for the top-scoring classes.
    mean_scores = np.mean(scores, axis=0)
    top_N = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_N]

    plt.subplot(3, 1, 3)
    scores_top_class = scores[:, top_class_indices].T
    # https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
    extent = (0, scores_top_class.shape[1] * YAMNetParams.PATCH_HOP_SECONDS, scores_top_class.shape[0] - 0.5, -0.5)
    plt.imshow(scores_top_class, aspect='auto', interpolation='nearest', cmap='gray_r', extent=extent)
    # TODO Compensate for the PATCH_WINDOW_SECONDS (0.96 s) context window to align with spectrogram.
    # patch_padding = (params.PATCH_WINDOW_SECONDS / 2) / params.PATCH_HOP_SECONDS
    # plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
    plt.xlim([0, len(waveform) / sr])

    # Label the top_N classes.
    yticks = range(0, top_N, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    _ = plt.ylim(-0.5 + np.array([top_N, 0]))
    plt.show()
