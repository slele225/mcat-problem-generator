"""Load and validate configuration from config.yaml."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DiscreteConfig:
    questions_per_topic: int = 50
    max_retries: int = 3
    temperature_generate: float = 0.8
    temperature_validate: float = 0.3
    batch_size: int = 20


@dataclass
class CARSConfig:
    passages_per_topic: int = 100
    questions_per_passage: int = 10
    passage_word_range: list = field(default_factory=lambda: [500, 600])
    max_retries: int = 3
    temperature_generate: float = 0.8
    temperature_validate: float = 0.3
    batch_size: int = 10
    passage_subjects: list = field(default_factory=list)


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-32B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    discrete: DiscreteConfig = field(default_factory=DiscreteConfig)
    cars: CARSConfig = field(default_factory=CARSConfig)
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./output"
    topics_file: str = "./topics.json"


def load_config(path: str = "config.yaml") -> Config:
    """Load config from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    discrete = DiscreteConfig(**{k: v for k, v in raw.get("discrete", {}).items()
                                  if k in DiscreteConfig.__dataclass_fields__})
    cars = CARSConfig(**{k: v for k, v in raw.get("cars", {}).items()
                         if k in CARSConfig.__dataclass_fields__})

    config = Config(
        model=raw.get("model", Config.model),
        vllm_base_url=raw.get("vllm_base_url", Config.vllm_base_url),
        discrete=discrete,
        cars=cars,
        checkpoint_dir=raw.get("checkpoint_dir", Config.checkpoint_dir),
        output_dir=raw.get("output_dir", Config.output_dir),
        topics_file=raw.get("topics_file", Config.topics_file),
    )

    # Ensure directories exist
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    return config
