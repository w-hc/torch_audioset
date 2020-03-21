'''
Conform that torchaudio dataloading is consistent with PyAV and soundfile.
Is PyAV slower than torchaudio? Play around and find out
'''
import numpy as np
import torchaudio as ta
import soundfile as sf
import librosa as la
import av


def ta_load(fname):
    x, sr = ta.load(fname)
    x = x.numpy()
    return x, sr


def sf_load_from_float(fname):
    x, sr = sf.read(fname, dtype='float32', always_2d=True)
    x = x.T
    return x, sr


def sf_load_from_int16(fname):
    x, sr = sf.read(fname, dtype='int16', always_2d=True)
    x = x / 2 ** 15
    x = x.T.astype(np.float32)
    return x, sr


def av_load(fname):
    container = av.open(fname)
    astream = container.streams.audio[0]
    num_channels = astream.codec_context.channels
    sr = astream.sample_rate
    frames = {}
    for audio_frame in container.decode(audio=0):
        frames[audio_frame.pts] = audio_frame
    frames = [ frames[i] for i in sorted(frames) ]
    data = np.concatenate(
        [af.to_ndarray().reshape(num_channels, -1) for af in frames], axis=1
    )
    container.close()
    if data.dtype == np.int16:
        assert data.dtype == np.int16
        data = data / (1. + np.iinfo(np.int16).max)
        data = data.astype(np.float32)
    assert data.dtype == np.float32
    return data, sr


def la_load(fname):
    x, sr = la.load(fname)
    x = x.T
    return x, sr


def test_sf_internal_consistency(fname):
    x, sr = sf_load_from_float(fname)
    y, sr = sf_load_from_int16(fname)
    assert np.allclose(x, y)


def test_ta_sf_consistency(fname):
    ta_data, sr = ta_load(fname)
    sf_data, sr = sf_load_from_float(fname)
    assert np.allclose(ta_data, sf_data)


def test_ta_av_consistency(fname):
    ta_data, sr = ta_load(fname)
    av_data, sr = av_load(fname)
    assert np.allclose(ta_data, av_data)


def test_av_la_consistency(fname):
    la_data, sr = la_load(fname)
    av_data, sr = av_load(fname)
    assert np.allclose(av_data, la_data)


def main():
    fname = '/home-nfs/whc/siren.wav'
    test_sf_internal_consistency(fname)
    test_ta_sf_consistency(fname)
    # BUG torchaudio and PyAV disagree, which means that your audio visual processing
    # pipeline could all be wrong!!!
    test_ta_av_consistency(fname)

    fname = '/home-nfs/whc/tv_host.mp4'
    test_av_la_consistency(fname)  # BUG completely bad! Even the sr do not agree!


if __name__ == "__main__":
    main()
