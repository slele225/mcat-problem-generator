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
    # Independent model used ONLY for the discrete blind-solve check (generation +
    # adversarial review stay on the main `model`). Routed to Sonnet — like CARS —
    # for more reliable JSON-following and stronger performance on hard items than
    # the Haiku default; it is in MODEL_PRICING so cost attributes correctly.
    discrete_checker_model: str = "claude-sonnet-4-6"


@dataclass
class CARSConfig:
    passages_per_topic: int = 100  # TOTAL passages to generate (not per-category)
    questions_per_passage: int = 10  # fixed fallback, used only when _range is unset
    # When non-empty, questions per passage are drawn uniformly from this inclusive
    # range each passage (mirrors science_passage; real CARS carries ~5-7), and
    # questions_per_passage above is ignored.
    questions_per_passage_range: list = field(default_factory=list)
    passage_word_range: list = field(default_factory=lambda: [500, 600])
    max_retries: int = 3
    temperature_generate: float = 0.8
    temperature_validate: float = 0.3
    batch_size: int = 10
    humanities_subjects: list = field(default_factory=list)
    social_science_subjects: list = field(default_factory=list)
    # Independent (and cheaper-than-Opus) model used ONLY for the CARS blind-solve
    # check, so a different model checks the generator rather than Opus checking
    # Opus. Generation and adversarial review stay on the main `model`. Sonnet is
    # strong enough for dense CARS reading; it is in MODEL_PRICING so cost is
    # attributed correctly.
    cars_checker_model: str = "claude-sonnet-4-6"


@dataclass
class SciencePassageConfig:
    # How many passages to generate per topic cluster.
    passages_per_topic_cluster: int = 1
    # Questions per passage are drawn uniformly from this inclusive range each
    # passage (real MCAT science passages carry 4-7 linked questions).
    questions_per_passage_range: list = field(default_factory=lambda: [4, 7])
    # Cluster sizing: aim for 2-3 related topics per passage.
    min_topics_per_cluster: int = 2
    max_topics_per_cluster: int = 3
    passage_word_range: list = field(default_factory=lambda: [200, 350])
    # Per-passage skill mix, weighted toward Skills 2/3/4 (vs. discrete 35/45/10/10).
    skill_weights: dict = field(default_factory=lambda: {
        "skill_1": 15, "skill_2": 35, "skill_3": 25, "skill_4": 25
    })
    max_retries: int = 3
    temperature_generate: float = 0.8
    temperature_validate: float = 0.3
    batch_size: int = 5
    # Independent model used ONLY for the science blind-solve check (generation +
    # adversarial review stay on the main `model`). Routed to Sonnet — like CARS —
    # for more reliable JSON-following and stronger performance on hard items than
    # the Haiku default; it is in MODEL_PRICING so cost attributes correctly.
    science_checker_model: str = "claude-sonnet-4-6"
    # Figure support (chemistry SMILES + matplotlib plots). When False, the
    # generation prompts do not offer figures and any stray spec is stripped, so
    # rdkit/matplotlib are never imported. When True, those deps are required
    # (checked at startup). Rendering is a separate `--render-figures` pass.
    enable_figures: bool = False
    figure_format: str = "png"  # "png" or "svg"


@dataclass
class Config:
    model: str = "claude-sonnet-4-6"
    # Cheaper model used only for the blind-solve validation step, so it acts as
    # an independent (and lower-cost) checker. Generation and adversarial review
    # always use `model`.
    checker_model: str = "claude-haiku-4-5-20251001"
    discrete: DiscreteConfig = field(default_factory=DiscreteConfig)
    cars: CARSConfig = field(default_factory=CARSConfig)
    science_passage: SciencePassageConfig = field(default_factory=SciencePassageConfig)
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
    science_passage = SciencePassageConfig(
        **{k: v for k, v in raw.get("science_passage", {}).items()
           if k in SciencePassageConfig.__dataclass_fields__})

    config = Config(
        model=raw.get("model", Config.model),
        checker_model=raw.get("checker_model", Config.checker_model),
        discrete=discrete,
        cars=cars,
        science_passage=science_passage,
        checkpoint_dir=raw.get("checkpoint_dir", Config.checkpoint_dir),
        output_dir=raw.get("output_dir", Config.output_dir),
        topics_file=raw.get("topics_file", Config.topics_file),
    )

    # Ensure directories exist
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    return config
