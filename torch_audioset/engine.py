import logging
import os
import os.path as osp
import json
import torch
import numpy as np
from .params import CommonParams
from .utils.logger import setup_logger
from .utils import comm
from .evaluator import inference_on_dataset, DatasetEvaluator
# from .vggish import get_vggish, vggish_category_metadata
from .yamnet import yamnet, yamnet_category_metadata
from .data.torch_input_processing import WaveformToInput

import itertools
from collections import OrderedDict


'''
General labeling strategy:
Label each audio one-by-one, for all the chunks, and save them inside what not
'''


def setup():
    """
    Perform some basic common setups at the beginning of a job. For now, only:
    1. Set up the logger
    """
    rank = comm.get_rank()
    logger = logging.getLogger(__name__)
    if not logger.isEnabledFor(logging.INFO):
        setup_logger(distributed_rank=rank)


def classify_audio_dataset(dataset, output_dir):
    # 0. preliminary setup
    setup()

    # 1. create model
    model = AudioLabeler(
        model=yamnet(pretrained=True),
        tt_chunk_size=CommonParams.VGGISH_CHUNK_SIZE
    )
    pred_category_meta = yamnet_category_metadata()

    # 2. create data loader
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=CommonParams.NUM_LOADERS,
        batch_size=1, collate_fn=trivial_collate_fn,
    )

    # 3. create evaluator
    evaluator = SoundLabelingEvaluator(output_dir, pred_category_meta)

    # 4. launch inference
    inference_on_dataset(model, loader, evaluator)


def trivial_collate_fn(inputs):
    return inputs


class AudioLabeler(torch.nn.Module):
    def __init__(self, model, tt_chunk_size):
        super().__init__()
        self.model = model
        self.tt_chunk_size = tt_chunk_size
        self.device = torch.device('cuda')
        self.data_transform = WaveformToInput()
        self.to(self.device)

    def forward(self, inputs):
        # inference context is setup by caller
        assert len(inputs) == 1
        overall_preds = []
        for payload in inputs:
            data, sr = payload['data'], payload['sr']
            # almost an hour for a segment
            num_frames_per_segment = int(3600 * CommonParams.PATCH_WINDOW_IN_SECONDS * sr)
            hourly_segments = data.split(num_frames_per_segment, dim=1)
            hourly_accu = []
            for segment in hourly_segments:
                segment = segment.to(self.device)
                segment = self.data_transform(segment, sr)
                chunks = segment.split(self.tt_chunk_size, dim=0)
                accu = []
                for chu in chunks:
                    pred = self.model(chu)  # [chunk_size, num_cats]
                    accu.append(pred.cpu())
                accu = torch.cat(accu, dim=0)
                hourly_accu.append(accu)
            hourly_accu = torch.cat(hourly_accu, dim=0)  # [num_hours, num_chunks, ]
            overall_preds.append({
                'pred_tsr': hourly_accu,
            })
        return overall_preds


class SoundLabelingEvaluator(DatasetEvaluator):
    """Labeling every chunk unit of a sound track
    It saves prediction in `output_dir`

    Output metadata format:
    {
        model: vggish,
        model_categories: [
            {
                name: specch,
                id: 3759347
            },
            ...
        ],
        predictions: [
            {
                id: 12345,
                category_tsr_fname: 12345.npy
                per_chunk_length: 0.96 (in seconds)
                meta: {} an optional payload copied verbatim from the inputs
            }
            ...
        ]
    }
    """
    TSR_DUMP_DIR = 'audio_pred'
    PRED_FNAME = 'audio_pred.json'

    def __init__(self, output_dir, pred_category_meta):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.pred_category_meta = pred_category_meta
        os.makedirs(osp.join(self.output_dir, self.TSR_DUMP_DIR), exist_ok=True)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for in_payload, out_payload in zip(inputs, outputs):
            # 1. save the audio category tensor
            audio_id = in_payload['id']
            category_tsr_fname = '{}.npy'.format(audio_id)
            np.save(
                osp.join(self.output_dir, self.TSR_DUMP_DIR, category_tsr_fname),
                out_payload['pred_tsr'].numpy()
            )
            # 2. record the prediction metadata
            pred = {
                'id': audio_id,
                'category_tsr_fname': category_tsr_fname,
                # 'scanned_segment': '',
                'per_chunk_length': CommonParams.PATCH_WINDOW_IN_SECONDS,
                'meta': '' if 'meta' not in in_payload else in_payload['meta']
            }
            self._predictions.append(pred)

    def evaluate(self):
        comm.synchronize()
        predictions = comm.gather(self._predictions)
        predictions = list(itertools.chain(*predictions))
        if not comm.is_main_process():
            return None

        payload = {
            'model': 'vggish',
            'model_categories': self.pred_category_meta,
            'predictions': predictions
        }

        with open(osp.join(self.output_dir, self.PRED_FNAME), 'w') as f:
            json.dump(payload, f)
