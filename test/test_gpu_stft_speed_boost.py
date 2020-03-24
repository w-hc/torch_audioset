import numpy as np
import torch
from torch_audioset.data.torch_input_processing import WaveformToInput
from fabric.utils.timer import Timer


class Test():
    def __init__(self, num_hours):
        self.data, self.sr = self.get_data(num_hours)

    def process(self, device_tag):
        device = torch.device(device_tag)
        timer = Timer()
        data = torch.from_numpy(self.data).to(device)
        trans = WaveformToInput().to(device)

        timer.tic()
        data = trans(data, self.sr)
        timer.toc()
        print("time taken for {} {}".format(device_tag, timer.avg))

    @staticmethod
    def get_data(num_hours):
        num_secs = num_hours * 60 * 60
        freq = 1000
        sample_rate = 44100
        t = np.linspace(0, num_secs, int(num_secs * sample_rate))
        x = np.sin(2 * np.pi * freq * t)
        # float32, [C=2, T=num_secs * sr]
        x = x.astype(np.float32).reshape(1, -1)
        return x, sample_rate


def main():
    # one extra in front to warm up the GPU
    for num_hours in [0.1, 0.1, 0.5, 1.0, 2.0]:
        print('num hours of mono audio {}'.format(num_hours))
        tester = Test(num_hours)
        tester.process('cpu')
        tester.process('cuda')


if __name__ == "__main__":
    main()


'''
GPU is indeed orders of magnitude faster. But it cannot process anything
longer than 2 hours due to memory constraints

num hours of mono audio 0.1
time taken for cpu 4.203256719978526
time taken for cuda 0.0498933190247044
num hours of mono audio 0.5
time taken for cpu 21.335003347019665
time taken for cuda 0.08045920997392386
num hours of mono audio 1.0
time taken for cpu 42.78843626496382
time taken for cuda 0.1552953750360757
num hours of mono audio 2.0
time taken for cpu 84.52113410702441
time taken for cuda 1.3808632729342207
'''
