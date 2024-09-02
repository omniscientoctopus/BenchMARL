"""
This module is the interface between the CustomEnv class and the BenchMARL environment module.


For documentation/reference: https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/environments/common.py

"""

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from .env import CustomEnv


class CustomTask(Task):

    # Your task names.
    # Their config will be loaded from conf/task/custom

    TOY = None  # Loaded automatically from conf/task/custom/toy

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: CustomEnv(
            categorical_actions=True, seed=seed, device=device, **self.config
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        NotImplementedError

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        NotImplementedError  # ToDo

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        NotImplementedError  # ToDo

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        NotImplementedError  # ToDo

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        NotImplementedError  # ToDo

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        NotImplementedError  # ToDo

    @staticmethod
    def env_name() -> str:
        return "custom_env"
