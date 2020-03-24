import os
import os.path as osp
import numpy as np
import torch
from torch_audioset.engine import classify_audio_dataset


# create a dummy dataset
class DummyDset(torch.utils.data.Dataset):
    def __init__(self):
        self.size = 100

    @staticmethod
    def get_data():
        num_secs = 1.5 * 60 * 60
        freq = 1000
        sample_rate = 44100
        t = np.linspace(0, num_secs, int(num_secs * sample_rate))
        x = np.sin(2 * np.pi * freq * t)
        # float32, [C=2, T=num_secs * sr]
        x = x.astype(np.float32).reshape(1, -1)
        return x, sample_rate

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, sr = self.get_data()
        x = torch.from_numpy(x).float()
        return {
            'id': str(index),
            'data': x, 'sr': sr,
            'meta': {
                'secret': 'top-secret'
            }
        }


def main():
    # create output dir
    this_file_dir = osp.abspath(osp.dirname(__file__))
    output_dir = osp.join(this_file_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    dset = DummyDset()
    classify_audio_dataset(dset, output_dir)


if __name__ == "__main__":
    main()
