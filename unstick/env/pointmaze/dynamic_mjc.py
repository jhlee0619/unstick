# From D4RL 1.1 (d4rl/pointmaze/dynamic_mjc.py), Apache License 2.0 (LICENSES/Apache-2.0.txt); modified.
"""A small library for programmatically building MuJoCo XML files."""
from contextlib import contextmanager
import tempfile
import numpy as np


class MJCModel(object):
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val).lower() for val in value])
        else:
            value = str(value).lower()

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items():
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))
