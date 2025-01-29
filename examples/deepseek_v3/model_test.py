# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================å
"""Tests for the DeepSeek model."""


from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import model as model_lib
import model_pytorch as model_lib_pt
import jax
import jax.numpy as jnp
import numpy as np
import torch


def bf16_pt_to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.array(
      pt_tensor.detach().to(torch.float32).numpy().astype(jnp.bfloat16)
  )

def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


class ModelTest(parameterized.TestCase):
  def test_rsnorm_parity(self):
    """Tests that the RSNorm layer is implemented correctly."""
    hidden_size = 16

    # pytorch
    model_pt = model_lib_pt.DeepseekV3RMSNorm(hidden_size)
    x_pt = torch.randn(1, 5, hidden_size)
    y_pt = model_pt(x_pt)

    # jax
    model_jax = model_lib.RMSNorm(hidden_size)
    model_jax.weight.value = to_jax(model_pt.weight)
    x_jax = to_jax(x_pt)
    y_jax = model_jax(x_jax)

    np.testing.assert_allclose(to_jax(y_pt), y_jax, atol=1e-3)

  @parameterized.parameters(
    (
      model_lib.RotaryEmbedding,
      model_lib_pt.DeepseekV3RotaryEmbedding,
    ),
    (
      model_lib.LinearScalingRotaryEmbedding,
      model_lib_pt.DeepseekV3LinearScalingRotaryEmbedding,
    ),
    (
      model_lib.DynamicNTKScalingRotaryEmbedding,
      model_lib_pt.DeepseekV3DynamicNTKScalingRotaryEmbedding,
    ),
    (
      model_lib.YarnRotaryEmbedding,
      model_lib_pt.DeepseekV3YarnRotaryEmbedding,
    ),
  )
  def test_rotary_embedding(
    self,
    rotary_type_jax: type[model_lib.RotaryEmbedding],
    rotary_type_pt: type[model_lib_pt.DeepseekV3RotaryEmbedding],
  ):
    hidden_dim = 32
    seq_len = 8

    # pytorch
    model_pt = rotary_type_pt(
      dim=hidden_dim, max_position_embeddings=128, base=10000
    )
    x_pt = torch.randn(2, 4, seq_len, hidden_dim)
    cos_pt, sin_pt = model_pt(x_pt, seq_len=seq_len)

    # jax
    model_jax = rotary_type_jax(hidden_dim, 128, 10000)
    model_jax.inv_freq.value = to_jax(model_pt.inv_freq)
    model_jax.cos_cached.value = to_jax(model_pt.cos_cached)
    model_jax.sin_cached.value = to_jax(model_pt.sin_cached)
    x_jax = to_jax(x_pt)
    cos_jax, sin_jax = model_jax(x_jax, seq_len=seq_len)

    np.testing.assert_allclose(to_jax(cos_pt), cos_jax, atol=1e-3)
    np.testing.assert_allclose(to_jax(sin_pt), sin_jax, atol=1e-3)

  def test_mlp_parity(self):
    """Tests that the MLP layer is implemented correctly."""
    hidden_size = 16
    mlp_dim = 32
    config = model_lib.Config()

    # pytorch
    model_pt = model_lib_pt.DeepseekV3MLP(config, hidden_size, mlp_dim)
    x_pt = torch.randn(1, 5, hidden_size)
    y_pt = model_pt(x_pt)

    # jax
    model_jax = model_lib.MLP(config, hidden_size, mlp_dim, rngs=nnx.Rngs(0))
    model_jax.gate_proj.kernel.value = to_jax(model_pt.gate_proj.weight).T
    model_jax.up_proj.kernel.value = to_jax(model_pt.up_proj.weight).T
    model_jax.down_proj.kernel.value = to_jax(model_pt.down_proj.weight).T
    x_jax = to_jax(x_pt)
    y_jax = model_jax(x_jax)

    np.testing.assert_allclose(to_jax(y_pt), y_jax, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
