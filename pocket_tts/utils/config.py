"""Configuration models for loading YAML config files."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from pocket_tts.default_parameters import DEFAULT_EOS_THRESHOLD

CONFIGS_DIR = Path(__file__).parent.parent / "config"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    dim: int
    depth: int
    type: Literal["lsd", "drifting"] = "lsd"


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int
    attn_dim: int | None = None
    num_kv_heads: int | None = None


class LookupTable(StrictModel):
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str
    rank: int | None = None


# Root configuration
class FlowLMConfig(StrictModel):
    """Root configuration model for YAML config files."""

    dtype: str

    # Nested configurations
    flow: FlowConfig
    transformer: FlowLMTransformerConfig

    # conditioning
    lookup_table: LookupTable
    weights_path: str | None = None
    insert_bos_before_voice: bool = False
    z_dim: int | None = None


# SEANet configuration
class SEANetConfig(StrictModel):
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int


# Transformer configuration for Mimi
class MimiTransformerConfig(StrictModel):
    d_model: int
    input_dimension: int
    output_dimensions: tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    max_period: float = 10000.0
    dim_feedforward: int


# Quantizer configuration
class QuantizerConfig(StrictModel):
    dimension: int
    output_dimension: int


# Root configuration
class MimiConfig(StrictModel):
    """Root configuration model for Mimi YAML config files."""

    dtype: str

    # Sample rate and channels
    sample_rate: int
    channels: int
    frame_rate: float

    # SEANet configurations
    seanet: SEANetConfig

    # Transformer
    transformer: MimiTransformerConfig

    # Quantizer
    quantizer: QuantizerConfig
    weights_path: str | None = None
    inner_dim: int | None = None
    outer_dim: int | None = None
    decoder_seanet: SEANetConfig | None = None
    decoder_transformer: MimiTransformerConfig | None = None


class Config(StrictModel):
    default_temperature: float = 0.3
    default_eos_threshold: float = DEFAULT_EOS_THRESHOLD
    replace_characters: dict[str, str] = {}
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: str | None = None
    weights_path_without_voice_cloning: str | None = None
    pad_with_spaces_for_short_inputs: bool = False
    remove_semicolons: bool = False
    model_recommended_frames_after_eos: int | None = None

    @model_validator(mode="after")
    def _reject_unimplemented_options(self) -> Self:
        # Accepted so that upstream configs load unchanged, but no model code reads them.
        unimplemented = [
            name
            for name, value in (
                ("flow_lm.transformer.attn_dim", self.flow_lm.transformer.attn_dim),
                ("flow_lm.transformer.num_kv_heads", self.flow_lm.transformer.num_kv_heads),
                ("flow_lm.lookup_table.rank", self.flow_lm.lookup_table.rank),
            )
            if value is not None
        ]
        if unimplemented:
            raise ValueError(f"This exporter does not implement: {', '.join(unimplemented)}")
        return self


def load_config(yaml_path: str | Path) -> Config:
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        if yaml_path.is_relative_to(CONFIGS_DIR):
            available = [path.stem for path in CONFIGS_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"Config file not found: {yaml_path}. "
                f"Did you make a typo? Available languages: {available}"
            )
        raise FileNotFoundError(f"Config file not found: {yaml_path}. Did you make a typo?")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
