import os
import os.path as osp
import json
from typing import List
from ..utils import mapify_iterable

__all__ = ['AudioSetOntology', ]


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
    def __init__(self, meta):
        self.meta = meta
        self.id = meta['id']
        self.name = meta['name']
        self.is_concrete = 'abstract' not in meta['restrictions']
        self.children = dict()
        self.parent = None

    def __repr__(self):
        s = 'name: {}; '.format(self.name)
        s += 'children: {}'.format(self.display_children_info(print_out=False))
        return s

    def display_children_info(self, print_out=True):
        accu = []
        for name, node in self.children.items():
            accu.append(name)
        if print_out:
            print(accu)
        else:
            return accu

    def add_child(self, c_node):
        self.children[c_node.name] = c_node

    def get_child(self, c_name):
        return self.children[c_name]

    @classmethod
    def trace(cls, node, trace: List):
        assert isinstance(trace, list)
        if len(trace) == 0:
            return node
        else:
            key = trace[0]
            child = node.children[key]
            return cls.trace(child, trace[1:])

    @classmethod
    def get_non_abstract_from_below(cls, start_node):
        accu = []

        def traverse(node):
            if node.is_concrete:
                accu.append(node)
            for name, child in node.children.items():
                traverse(child)

        traverse(start_node)
        return accu


class AudioSetOntology():
    def __init__(self):
        dir_of_this_mod = osp.dirname(osp.abspath(__file__))
        fname = osp.join(dir_of_this_mod, 'ontology.json')
        with open(fname, 'r') as f:
            raw = json.load(f)
        self.raw = mapify_iterable(raw, 'id')
        self.tree = self.build_tree(self.raw)

    @staticmethod
    def build_tree(id_2_meta):
        tree_accu = dict()
        candidates = list(id_2_meta.keys())

        def create(id):
            '''recursively build up an ontology tree'''
            if id in tree_accu:
                return tree_accu[id]
            meta = id_2_meta[id]
            node = TreeNode(meta)
            child_ids = meta['child_ids']
            for cid in child_ids:
                c_node = create(cid)
                c_node.parent = node
                node.add_child(c_node)
            tree_accu[id] = node
            return node

        for c in candidates:
            create(c)

        # put those nodes without parents under the root node
        rootNode = TreeNode(
            meta={'id': 'root', 'name': 'root', 'restrictions': ['abstract']}
        )
        for _, node in tree_accu.items():
            if node.parent is None:
                rootNode.add_child(node)
                node.parent = rootNode

        return rootNode

    def query(self, tracer_list: List[str]):
        pass
