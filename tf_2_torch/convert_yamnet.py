from collections import OrderedDict
from itertools import chain
import torch
import numpy as np
import h5py
from h5py import Dataset
import tensorflow as tf
from torch_audioset.yamnet.model import yamnet as torch_yamnet
import params as tf_params


class TFYamn():
    def __init__(self, ckpt_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            yamnet = self.yamnet_tf_model()
            yamnet.load_weights(ckpt_path)
            self.net = yamnet

    @staticmethod
    def yamnet_tf_model():
        from yamnet import (
            layers, Model, features_lib, yamnet
        )
        waveform = layers.Input(batch_shape=(1, None))
        # Store the intermediate spectrogram features to use in visualization.
        spectrogram = features_lib.waveform_to_log_mel_spectrogram(
            tf.squeeze(waveform, axis=0), tf_params)
        patches = features_lib.spectrogram_to_patches(spectrogram, tf_params)
        predictions = yamnet(patches)
        # only difference is here
        frames_model = Model(
            name='yamnet_frames',
            inputs=waveform,
            outputs=[spectrogram, patches, predictions]
        )
        return frames_model

    def __call__(self, x):
        with self.graph.as_default():
            return self.net.predict(x, steps=1)


def load_keras_weight_in_pt_format(path):
    yamnet_weight_ordered_keys = [
        [
            'layer1/conv/layer1/conv/kernel:0',
            'layer1/conv/bn/layer1/conv/bn/beta:0',
            'layer1/conv/bn/layer1/conv/bn/moving_mean:0',
            'layer1/conv/bn/layer1/conv/bn/moving_variance:0'
        ],
        # layer [2 - 14]
        [
            'layer_i/depthwise_conv/layer_i/depthwise_conv/depthwise_kernel:0',
            'layer_i/depthwise_conv/bn/layer_i/depthwise_conv/bn/beta:0',
            'layer_i/depthwise_conv/bn/layer_i/depthwise_conv/bn/moving_mean:0',
            'layer_i/depthwise_conv/bn/layer_i/depthwise_conv/bn/moving_variance:0',
            'layer_i/pointwise_conv/layer_i/pointwise_conv/kernel:0',
            'layer_i/pointwise_conv/bn/layer_i/pointwise_conv/bn/beta:0',
            'layer_i/pointwise_conv/bn/layer_i/pointwise_conv/bn/moving_mean:0',
            'layer_i/pointwise_conv/bn/layer_i/pointwise_conv/bn/moving_variance:0',
        ],
        [
            'logits/logits/kernel:0',
            'logits/logits/bias:0',
        ],
    ]
    # 1. get the right loading order
    ordered_keys = [yamnet_weight_ordered_keys[0], ]
    layer_keys = yamnet_weight_ordered_keys[1]
    for i in range(2, 15):
        layer_i_keys = [ k.replace('layer_i', 'layer{}'.format(i)) for k in layer_keys ]
        ordered_keys.append(layer_i_keys)
    ordered_keys.append(yamnet_weight_ordered_keys[2])
    ordered_keys = list(chain(*ordered_keys))
    param_dict = OrderedDict()

    # 2. load the weights, and convert shape to pt format
    ckpt = h5py.File(path, 'r')
    for name in ordered_keys:
        weight = ckpt[name][:]
        if len(weight.shape) == 4:  # conv kernel
            # [H, W, c_in, c_out] -> [c_out, c_in, H, W]
            weight = weight.transpose(3, 2, 0, 1)
            if 'depthwise' in name and weight.shape[0] == 1:  # grouped conv
                weight = weight.transpose(1, 0, 2, 3)
        elif len(weight.shape) == 2:  # FC
            weight = weight.transpose()
        param_dict[name] = weight

    # 3. confirm that no weights are left out
    accu = {}
    def put_weight(name, object):
        if isinstance(object, Dataset):
            accu[name] = Dataset
    ckpt.visititems(put_weight)
    assert len(accu) == len(param_dict)

    return param_dict


def load_tf_weights(pt_state_dict, tf_state_dict):
    # 1. filter out weights in pt model that doesn't need loading
    old_pt_state_dict = pt_state_dict
    pt_state_dict = OrderedDict()
    for k, v in old_pt_state_dict.items():
        if 'bn.num_batches_tracked' in k or 'bn.weight' in k:
            continue
        assert v.numel() > 0
        pt_state_dict[k] = v

    # 2. match param and load
    for pt_k, tf_k in zip(pt_state_dict.keys(), tf_state_dict.keys()):
        print("matching {:<50} <--->     {}".format(pt_k, tf_k))
        assert pt_state_dict[pt_k].shape == tf_state_dict[tf_k].shape
        pt_state_dict[pt_k].copy_(
            torch.from_numpy(tf_state_dict[tf_k]).float()
        )


def main():
    np.random.seed(100)
    ckpt_path = 'yamnet.h5'
    tf_model = TFYamn(ckpt_path)
    # snd = np.random.uniform(-1.0, +1.0, (1, int(3 * tf_params.SAMPLE_RATE)))
    snd = np.sin(2 * np.pi * 440 * np.linspace(0, 3, int(3 * tf_params.SAMPLE_RATE)))
    snd = snd.reshape(1, -1)
    spec, patches, tf_pred = tf_model(snd)
    # spec: [298, 64], patches: [5, 96, 64]

    pt_model = torch_yamnet(pretrained=False)
    load_tf_weights(
        pt_model.state_dict(), load_keras_weight_in_pt_format(ckpt_path)
    )

    with torch.no_grad():
        pt_model.eval()
        x = torch.from_numpy(patches)
        x = x.unsqueeze(1)  # [5, 96, 64] -> [5, 1, 96, 64]
        pt_pred = pt_model(x, to_prob=True)
        pt_pred = pt_pred.numpy()

    print(pt_pred.shape)
    print(tf_pred.shape)

    print(pt_pred.mean(), pt_pred.std())
    print(tf_pred.mean(), tf_pred.std())

    assert pt_pred.shape == tf_pred.shape
    assert np.allclose(pt_pred, tf_pred, atol=1e-6)
    print('conversion succeed')
    torch.save(pt_model.state_dict(), './yamnet.pth')


if __name__ == "__main__":
    main()
