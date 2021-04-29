'''
hyperparams for VGGish and YAMNet, plus common configs
vggish params are retrieved from:
    https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/audioset/vggish/vggish_params.py
yamnet params are retrieved from:
    https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/audioset/yamnet/params.py
'''


class CommonParams():
    # for STFT
    TARGET_SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010

    # for log mel spectrogram
    NUM_MEL_BANDS = 64
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.001  # NOTE 0.01 for vggish, and 0.001 for yamnet

    # convert input audio to segments
    PATCH_WINDOW_IN_SECONDS = 0.96

    # largest feedforward chunk size at test time
    VGGISH_CHUNK_SIZE = 128
    YAMNET_CHUNK_SIZE = 256

    # num of data loading threads
    NUM_LOADERS = 4


class VGGishParams():
    # Copyright 2017 The TensorFlow Authors All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    """Global parameters for the VGGish model.
    See vggish_slim.py for more information.
    """

    # Architectural constants.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.

    # Parameters used for embedding postprocessing.
    PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
    PCA_MEANS_NAME = 'pca_means'
    QUANTIZE_MIN_VAL = -2.0
    QUANTIZE_MAX_VAL = +2.0

    # Hyperparameters used in training.
    INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
    LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
    ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

    # Names of ops, tensors, and features.
    INPUT_OP_NAME = 'vggish/input_features'
    INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
    OUTPUT_OP_NAME = 'vggish/embedding'
    OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
    AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'


class YAMNetParams():
    # Copyright 2019 The TensorFlow Authors All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    """Hyperparameters for YAMNet."""

    # The following hyperparameters (except PATCH_HOP_SECONDS) were used to train YAMNet,
    # so expect some variability in performance if you change these. The patch hop can
    # be changed arbitrarily: a smaller hop should give you more patches from the same
    # clip and possibly better performance at a larger computational cost.
    SAMPLE_RATE = 16000
    STFT_WINDOW_SECONDS = 0.025
    STFT_HOP_SECONDS = 0.010
    MEL_BANDS = 64
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.001
    PATCH_WINDOW_SECONDS = 0.96
    PATCH_HOP_SECONDS = 0.48

    PATCH_FRAMES = int(round(PATCH_WINDOW_SECONDS / STFT_HOP_SECONDS))
    PATCH_BANDS = MEL_BANDS
    NUM_CLASSES = 521
    CONV_PADDING = 'same'
    BATCHNORM_CENTER = True
    BATCHNORM_SCALE = False
    BATCHNORM_EPSILON = 1e-4
    CLASSIFIER_ACTIVATION = 'sigmoid'

    FEATURES_LAYER_NAME = 'features'
    EXAMPLE_PREDICTIONS_LAYER_NAME = 'predictions'


# NOTE for our inference, don't need overlapping windows
# YAMNetParams.PATCH_HOP_SECONDS = YAMNetParams.PATCH_WINDOW_SECONDS
YAMNetParams.PATCH_HOP_SECONDS = 1.0
