import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import json
from typing import List

from .engine import SoundLabelingEvaluator  # require info on pred storage format
from .utils import mapify_iterable
from .audioset_ontology.ontology import TreeNode
from tqdm import tqdm

__all__ = ['AudioCategoryLabels']

_POOLING_FUNC = {
    'ave': F.avg_pool1d,
    'max': F.max_pool1d
}['max']


class AudioCategoryLabels():
    def __init__(self, root, pooling_window_size=5, stride=1):
        json_fname = osp.join(root, SoundLabelingEvaluator.PRED_FNAME)
        with open(json_fname, 'r') as f:
            meta = json.load(f)
        self.root = root
        self.model = meta['model']
        self.cats = meta['model_categories']
        self.preds = mapify_iterable(meta['predictions'], 'id')
        self.video_ids = list(sorted(self.preds.keys()))

        self.pooling_window_size = pooling_window_size
        self.stride = stride

        self.response_tsr = None
        self.all_video_starting_ticks = None

    def process(self):
        '''Combine pred tsrs from all clips into one giant tsr so that
        we can do global ranking and comparison
        '''
        combined_response_arr = []
        all_video_starting_ticks = []
        _curr = 0
        for vid in tqdm(self.video_ids):
            all_video_starting_ticks.append(_curr)
            tsr = self._load_pred_tsr(vid)
            tsr = self._process_single(tsr, self.pooling_window_size, self.stride)
            combined_response_arr.append(tsr)
            num_ticks_after_striding = len(tsr)
            _curr += num_ticks_after_striding
        self.response_tsr = np.concatenate(combined_response_arr, axis=0)
        self.all_video_starting_ticks = np.array(
            all_video_starting_ticks, dtype=int
        )
        return self.response_tsr

    def match_cats_to_model_pred_inds(self, candidates: List[TreeNode]):
        '''Each model has its unique output category mappings
        Match the supplied AudioSet Cat nodes to the prediction category indices
        '''
        classifier_output_names = [ c['name'] for c in self.cats ]
        matched_inds = []
        matched_candidates = []
        for c in candidates:
            name = c.name
            if name in classifier_output_names:
                inx = classifier_output_names.index(name)
                matched_candidates.append(c)
                matched_inds.append(inx)
            else:
                # print("cat {} not included in this model".format(name))
                pass
        return matched_candidates, matched_inds

    def global_tick_to_video_interval(self, global_tick):
        '''
        Given a global tick index on the combined tensor, tell us
        from which video and what interval this tick corresponds to
        '''
        video_id, video_tick = self._global_tick_to_video_tick(global_tick)
        interval = self._video_tick_to_time_interval(video_id, video_tick)
        return {
            'video_id': video_id,
            'interval': interval
        }

    @staticmethod
    def _process_single(tsr, pooling_window_size, stride):
        assert torch.cuda.is_available()
        padding = (pooling_window_size - 1) // 2
        tsr = torch.from_numpy(tsr).cuda()
        tsr = tsr.T.unsqueeze(dim=0)  # [T, K]  -> [1, K, T]
        tsr = _POOLING_FUNC(
            tsr, kernel_size=pooling_window_size, stride=stride, padding=padding
        )
        tsr = tsr.squeeze(0).T
        tsr = torch.sigmoid(tsr).cpu().numpy()
        return tsr

    def _load_pred_tsr(self, vid, logit_to_prob=False):
        tsr = np.load(osp.join(
                self.root, SoundLabelingEvaluator.TSR_DUMP_DIR,
                self.preds[vid]['category_tsr_fname']
        ))
        self.preds[vid]['scanned_num_chunks'] = len(tsr)
        if logit_to_prob:
            tsr = torch.sigmoid(torch.from_numpy(tsr)).numpy()
        return tsr

    def _global_tick_to_video_tick(self, global_tick: int):
        which_video = np.searchsorted(
            self.all_video_starting_ticks, global_tick, side='right'
        ) - 1
        video_id = self.video_ids[which_video]
        video_starting_tick = self.all_video_starting_ticks[which_video]
        video_tick = (global_tick - video_starting_tick) * self.stride
        return video_id, video_tick

    def _video_tick_to_time_interval(self, video_id, interval_center_tick):
        meta = self.preds[video_id]
        time_unit = meta['per_chunk_length']
        num_chunks = meta['scanned_num_chunks']
        window_size = self.pooling_window_size
        radius = (window_size - 1) // 2
        start = interval_center_tick - radius
        end = start + window_size
        start, end = max(0, start), min(num_chunks, end)
        start, end = start * time_unit, end * time_unit
        return start, end
