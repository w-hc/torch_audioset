import numpy as np
import torch
from torch_audioset.vggish import get_vggish
from torch_audioset.data.torch_input_processing import WaveformToInput as torch_trans
from torch_audioset.data.tflow_input_processing import WaveformToInput as tflow_trans


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


@torch.no_grad()
def test_embeddings_with_torch_input_processing():
    device = torch.device('cuda')

    model = get_vggish(with_classifier=False, pretrained=True)
    model.eval()
    model.to(device)

    x, sample_rate, expected_stats = sinusoidal_data()
    x = torch.from_numpy(x).reshape(1, -1).float()  # [C, L]
    x = x.to(device)

    trans = torch_trans().to(device)

    x = trans(x, sample_rate)

    embeddings = model(x)
    embeddings = embeddings.cpu().numpy()

    mean, std = np.mean(embeddings), np.std(embeddings)
    print('expected mean {} vs actual mean {}'.format(
        expected_stats['embedding_mean'], mean)
    )
    print('expected std {} vs actual std {}'.format(
        expected_stats['embedding_std'], std)
    )


@torch.no_grad()
def test_embeddings_with_tflow_input_processing():
    device = torch.device('cuda')

    model = get_vggish(with_classifier=False, pretrained=True)
    model.eval()
    model.to(device)
    x, sample_rate, expected_stats = sinusoidal_data()
    x = x.reshape(1, -1)  # [C, L]

    trans = tflow_trans()
    x = trans(x, sample_rate)

    x = x.to(device)

    embeddings = model(x)
    embeddings = embeddings.cpu().numpy()

    mean, std = np.mean(embeddings), np.std(embeddings)
    print('expected mean {} vs actual mean {}'.format(
        expected_stats['embedding_mean'], mean)
    )
    print('expected std {} vs actual std {}'.format(
        expected_stats['embedding_std'], std)
    )


if __name__ == "__main__":
    test_embeddings_with_tflow_input_processing()
