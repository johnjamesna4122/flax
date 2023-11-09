import dataclasses
import typing as tp
from typing import Any

from flax import linen
from flax.experimental.nnx.nnx import helpers
from flax.experimental.nnx.nnx import variables as variableslib
from flax.experimental.nnx.nnx.module import Module, ModuleDef
from flax.experimental.nnx.nnx.rnglib import Rngs
from flax.experimental.nnx.nnx.state import State

M = tp.TypeVar('M', bound=Module)


# Flax-like style is NNX
@dataclasses.dataclass
class Functional(tp.Generic[M]):
  module_type: tp.Type[M]
  moduledef: tp.Optional[ModuleDef[M]]
  args: tuple[tp.Any, ...]
  kwargs: dict[str, tp.Any]

  def init(self, *, rngs: tp.Optional[Rngs] = None) -> State:
    kwargs = {}
    if rngs is not None:
      kwargs['rngs'] = rngs
    module = self.module_type(*self.args, **self.kwargs, **kwargs)
    state, moduledef = module.split()
    self.moduledef = moduledef
    return state

  def apply(self, *states: tp.Any):
    assert self.moduledef is not None
    return self.moduledef.apply(*states)


def functional(cls: tp.Type[M]) -> tp.Callable[..., Functional[M]]:
  def _functional_constructor(*args: tp.Any, **kwargs: tp.Any) -> Functional[M]:
    return Functional(cls, None, args, kwargs)

  return _functional_constructor


class LinenWrapper(Module):
  def __init__(
    self,
    module: linen.Module,
    *args: tp.Any,
    rngs: tp.Optional[Rngs] = None,
    **kwargs: tp.Any,
  ):
    self.module = module

    _rngs = (
      {name: stream.key for name, stream in rngs._rngs.items()} if rngs else {}
    )
    # rename default to params
    if 'params' not in _rngs and 'default' in _rngs:
      _rngs['params'] = _rngs['default']
      del _rngs['default']

    variables = module.init(_rngs, *args, **kwargs)

    self.states = helpers.Dict(
      (collection, variableslib.variable_type(collection)(value))
      for collection, value in variables.items()
    )

  def __call__(
    self, *args: Any, rngs: tp.Optional[Rngs] = None, **kwargs: Any
  ) -> Any:
    _rngs = (
      {name: stream.key for name, stream in rngs._rngs.items()} if rngs else {}
    )

    variables = {collection: value for collection, value in self.states.items()}
    out = self.module.apply(variables, *args, rngs=_rngs, **kwargs)

    if kwargs.get('mutable', False) != False:
      out, updates = out
      for collection, value in updates.items():
        if collection in self.states:
          self.states[collection] = value
        else:
          self.states[collection] = variableslib.variable_type(collection)(
            value
          )

    return out


class NNXWrapper(linen.Module):
  ...
