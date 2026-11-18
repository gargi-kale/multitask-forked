"""Moved to multitask/train/probes.py; this alias keeps
`from hidden_label_probe import run_probes` working."""
import sys

from multitask.train import probes as _module

sys.modules[__name__] = _module
