'''
Implement the main logic of the environment here.


For reference: https://github.com/pytorch/rl/tree/main/torchrl/envs/custom

'''

from typing import Callable, Dict, List, Optional

import numpy as np

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase


class CustomEnv(EnvBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _step(cls, tensordict):
        NotImplementedError

    def _reset(self, tensordict):
        NotImplementedError # ToDo

    def _set_seed(self, seed: int):
        NotImplementedError # ToDo