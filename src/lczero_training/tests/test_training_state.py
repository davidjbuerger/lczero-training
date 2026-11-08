"""Tests for TrainingState construction."""

import jax
from google.protobuf import text_format

from lczero_training.training.state import TrainingState
from proto.model_config_pb2 import ModelConfig
from proto.training_config_pb2 import TrainingConfig

_MODEL_CONFIG = """
defaults { compute_dtype: F32 activation: ACTIVATION_MISH }
embedding { dense_size: 16 embedding_size: 32 dff: 32 }
encoder {
  num_blocks: 1 dff: 32 d_model: 32 heads: 2
  smolgen { hidden_channels: 4 hidden_size: 16 gen_size: 16 }
}
policy_head { name: "vanilla" embedding_size: 32 d_model: 32 }
value_head { name: "winner" num_channels: 8 }
movesleft_head { name: "main" num_channels: 8 }
"""

_TRAINING_CONFIG = """
lr_schedule { starting_step: 0 duration_steps: 0 lr: 0.001 }
optimizer { nadamw { beta_1: 0.9 beta_2: 0.98 epsilon: 1e-7 } }
"""


def test_new_from_config_swa_state_does_not_alias_model_state() -> None:
    """The jitted train step donates the state, so buffers must be distinct."""
    model_config = text_format.Parse(_MODEL_CONFIG, ModelConfig())
    training_config = text_format.Parse(_TRAINING_CONFIG, TrainingConfig())

    state = TrainingState.new_from_config(model_config, training_config)
    jit_state = state.jit_state
    assert jit_state.swa_state is not None

    # Donating the same buffer twice raises, which is what train_step hits.
    donate = jax.jit(lambda m, s: (m, s), donate_argnums=(0, 1))
    donate(jit_state.model_state, jit_state.swa_state)
