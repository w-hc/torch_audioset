import os
import os.path as osp
import json
from typing import List

'''
The ontology json file format

id:                 /m/0dgw9r,
name:Male           speech, man speaking,
description:        A description of the class in a few lines.
citation_uri:       Any text used as the basis for the description. e.g. a Wikipedia page
positive_examples:  YouTube URLs of positive examples
child_ids:          ids of children classes of this class
restrictions:       ['abstract', 'blacklist'] a list of optional tags.
                    'abstract': the class is purely a parent class
                    'blacklist': the class is ambiguous to annotate
'''


class TreeNode():
    def __init__(self, category):
        self.category = category
        self.children = []


class AudioSetOntology():
    def __init__(self):
        dir_of_this_mod = osp.dirname(osp.abspath(__file__))
        fname = osp.join(dir_of_this_mod, 'ontology.json')
        with open(fname, 'r') as f:
            self.raw = json.load(f)

        # non-abstract classes
        leaf_cats = []
        node_cats = []
        hard_cats = []
        for category in self.raw:
            restrictions = category['restrictions']
            if len(restrictions) == 0:
                leaf_cats.append(category)
            elif 'abstract' in restrictions:
                node_cats.append(category)
            else:
                assert 'blacklist' in restrictions
                hard_cats.append(category)
        print(len(leaf_cats))
        print(len(node_cats))
        print(len(hard_cats))
        size = len(leaf_cats) + len(node_cats) + len(hard_cats)

    def build_tree(self):
        pass

    def query(self, tracer_list: List[str]):
        pass


def compute_data_usage():
    num_secs = 900 * 3600
    num_bytes = bytes_over_length_in_secs(num_secs)
    MB = bytes_to(num_bytes, 'MB')
    GB = bytes_to(num_bytes, 'GB')
    print('{:.3f}MB and {:.3f}GB'.format(MB, GB))


def bytes_over_length_in_secs(num_secs):
    bytes_per_960ms = 527 * 4
    num_segments = num_secs / 0.960
    return num_segments * bytes_per_960ms


def bytes_to(num_bytes, unit='MB'):
    assert unit in ('MB', 'GB')
    degrees = {
        'MB': 2,
        'GB': 3
    }
    return num_bytes / (1024 ** degrees[unit])


if __name__ == "__main__":
    # m = AudioSetOntology()
    compute_data_usage()
