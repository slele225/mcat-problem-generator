# MCAT Question Generator

Generate MCAT practice questions using a local LLM (Qwen2.5-32B-Instruct) served via vLLM on an A100 GPU.

## Features

- **Discrete questions**: Standalone MCQs for Bio/Biochem, Chem/Phys, and Psych/Soc (50 per topic)
- **CARS passages**: 500-600 word humanities passages with 10 questions each
- **Two-stage validation**: Every question goes through adversarial review + blind solve
- **Checkpoint/resume**: Interrupt anytime with Ctrl+C, resume by running again
- **Configurable**: Counts, temperatures, batch sizes all in `config.yaml`

## Quick Start

SSH into your GPU server, clone/copy this project, then:

```bash
chmod +x run.sh
./run.sh
```

That's it. The script handles everything: installs [uv](https://docs.astral.sh/uv/), creates a venv, installs dependencies, starts vLLM, waits for the model to load, and runs the pipeline.

On first run it will download the Qwen2.5-32B-Instruct model (~20GB), so expect a few minutes of setup.

## Usage

```bash
# Run both pipelines (discrete + CARS)
./run.sh

# Run only discrete questions
./run.sh --discrete-only

# Run only CARS passages
./run.sh --cars-only

# Check progress
./run.sh --stats

# Verbose/debug logging
./run.sh -v

# Clear all checkpoints and start fresh
./run.sh --reset
```

## Manual Setup (if you prefer)

If you'd rather manage things yourself instead of using `run.sh`:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start vLLM in one terminal
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code

# Run the generator in another terminal
uv run python -m src.main
```

## Output

- `output/discrete_questions.jsonl` — One question per line
- `output/cars_passages.jsonl` — One passage (with its questions) per line

### Discrete question format

```json
{
  "question_id": "BB_1A_001_q003",
  "topic_id": "BB_1A_001",
  "section": "Biological and Biochemical Foundations of Living Systems",
  "content_category": "1A: Structure and function of proteins...",
  "topic": "Description",
  "stem": "A researcher observes that glycine...",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "B",
  "explanation": "...",
  "difficulty": "medium",
  "validation": {"adversarial_pass": true, "blind_solve_pass": true}
}
```

### CARS passage format

```json
{
  "passage_id": "CARS_P0042",
  "passage_text": "In the decades following...",
  "word_count": 547,
  "subject": "Philosophy",
  "questions": [
    {
      "question_id": "CARS_P0042_q01",
      "skill_type": "Foundations of Comprehension",
      "stem": "The author's primary purpose...",
      "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "C",
      "explanation": "..."
    }
  ]
}
```

## Configuration

Edit `config.yaml` to adjust generation parameters:

```yaml
model: Qwen/Qwen2.5-32B-Instruct
vllm_base_url: http://localhost:8000/v1

discrete:
  questions_per_topic: 50    # MCQs per topic (746 non-CARS topics)
  max_retries: 3             # Retries for failed validation
  temperature_generate: 0.8  # Higher = more varied questions
  temperature_validate: 0.3  # Lower = stricter validation
  batch_size: 20             # Concurrent requests to vLLM

cars:
  passages_per_topic: 100    # Total CARS passages to generate
  questions_per_passage: 10  # Questions per passage
  passage_word_range: [500, 600]
  batch_size: 10             # Lower — passages are longer
```

**Tip**: Start small to test quality before doing a full run:

```yaml
discrete:
  questions_per_topic: 5

cars:
  passages_per_topic: 3
```

## Resuming After Interruption

The checkpoint system tracks progress per-topic (discrete) and per-passage (CARS). Just run the same command again — it picks up where it left off.

```bash
./run.sh --stats   # See what's done
./run.sh           # Continue
```

## Project Structure

```
mcat-gen/
├── run.sh                    # One-shot setup + run script
├── config.yaml               # All tunable parameters
├── pyproject.toml            # uv/Python project config
├── topics.json               # MCAT topic data (your file)
├── src/
│   ├── main.py               # CLI entry point
│   ├── config.py             # Config loader
│   ├── llm_client.py         # Async vLLM client with batching
│   ├── schemas.py            # Pydantic models for output data
│   ├── checkpoint.py         # Checkpoint/resume system
│   ├── prompts/
│   │   ├── discrete.py       # Prompt templates for discrete Qs
│   │   └── cars.py           # Prompt templates for CARS
│   └── pipelines/
│       ├── discrete.py       # Discrete question pipeline
│       └── cars.py           # CARS passage pipeline
├── output/                   # Generated questions (JSONL)
├── checkpoints/              # Resume state
└── logs/                     # vLLM and generation logs
```

## Troubleshooting

**vLLM server crashes or OOMs**: Lower `gpu-memory-utilization` in `run.sh` (try 0.85) or reduce `max-model-len` to 2048.

**Questions are too easy / low quality**: Increase `temperature_generate` (try 0.9) and lower `temperature_validate` (try 0.2) in `config.yaml`. The stricter validation will reject more, but what passes will be better.

**Too many questions failing validation**: This is normal — the two-stage validation is intentionally strict. If your pass rate is below ~40%, try lowering `temperature_generate` to 0.7 so the model produces more "standard" questions.

**Model download is slow**: The Qwen2.5-32B model is ~20GB. If Hugging Face is slow, set `HF_HUB_ENABLE_HF_TRANSFER=1` and `pip install hf_transfer` for faster downloads.
