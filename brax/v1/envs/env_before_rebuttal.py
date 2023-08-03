# Copyright 2023 The Brax Authors.
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

"""A brax environment for training and inference."""

import abc
from typing import Any, Dict, Optional

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1 import pytree
from flax import struct

from google.protobuf import text_format


@struct.dataclass
class State:
  """Environment state for training and inference."""
  qp: brax.QP
  obs: jp.ndarray


@pytree.register
class Env(abc.ABC):
  """API for driving a brax system for training and inference."""

  """
  def __init__(self, config: Optional[str], *args, **kwargs):
    if config:
      config = text_format.Parse(config, brax.Config())
      self.sys = brax.System(config, *args, **kwargs)
  """

  def __init__(self, config_str=None,config_brax=None):
    if config_str is not None:
      config = text_format.Parse(config_str, brax.Config())
      self.sys = brax.System(config)
    elif config_brax is not None:
      self.sys = brax.System(config_brax)

  @abc.abstractmethod
  # def reset(self, rng: jp.ndarray) -> State:
  def reset(self, torso_pos: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

  @property
  def unwrapped(self) -> 'Env':
    return self


class Wrapper(Env):
  """Wraps the environment to allow modular transformations."""

  def __init__(self, env: Env):
    # super().__init__(config=None)
    super().__init__()
    self.env = env

  # def reset(self, rng: jp.ndarray) -> State:
  def reset(self, toros_pos: jp.ndarray) -> State:
    # return self.env.reset(rng)
    return self.env.reset(torso_pos)

  def step(self, state: State, action: jp.ndarray) -> State:
    return self.env.step(state, action)

  @property
  def unwrapped(self) -> Env:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)
