# review_bench

> Benchmark AI code reviewers. Pick the best one for your pipeline.

**review_bench** runs Claude, GPT-4, Gemini, and local Ollama models against a curated set of buggy code samples and produces precision/recall/F1 scorecards—so you can objectively compare models without relying on vendor-provided benchmarks. Results render as a Rich terminal table and export to JSON for CI or historical tracking.

---

## Quick Start

```bash
# Install
pip install review-bench

# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."

# Run the benchmark against one or more models
review-bench run --models gpt4o claude-sonnet

# Run against a local Ollama model (no API key needed)
review-bench run --models ollama:codellama

# Run all supported models
review-bench run --models gpt4o claude-sonnet gemini gemini ollama:llama3
```

After the run completes, a scorecard table prints to your terminal and full results are saved to a timestamped JSON file.

---

## Features

- **Multi-model support** — Plug in Claude (claude-3-opus/sonnet), GPT-4o, Gemini 1.5 Pro, or any local Ollama model through a unified adapter interface.
- **Standardized benchmark dataset** — 50+ buggy code samples across five categories: `null_dereference`, `off_by_one`, `sql_injection`, `race_condition`, and `resource_leak`, in Python, JavaScript, Java, and Go.
- **Precision/recall/F1 scoring** — Keyword and category matching against ground-truth labels gives a vendor-independent quality scorecard per model.
- **Side-by-side comparison table** — Rich-powered terminal table ranks all evaluated models by F1 score with per-category breakdowns.
- **Exportable results** — Full run results saved to timestamped JSON for CI integration, historical tracking, or sharing with your team.

---

## Usage Examples

### Run the benchmark

```bash
# Run against GPT-4o only
review-bench run --models gpt4o

# Run against multiple models and filter to a single bug category
review-bench run --models gpt4o claude-sonnet --category sql_injection

# Limit to 10 samples per category (faster iteration)
review-bench run --models ollama:codellama --max-samples 10

# Save results to a specific output directory
review-bench run --models gpt4o --output-dir ./results
```

### Browse benchmark samples

```bash
# List all samples
review-bench list-samples

# Filter by category
review-bench list-samples --category race_condition

# Filter by language
review-bench list-samples --language java
```

### Inspect a previous run

```bash
# Re-render a saved results file as a scorecard table
review-bench show-results results/run_20240601_143022.json
```

### Example scorecard output

```
                  Code Review Benchmark Results
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model           ┃ Precision ┃ Recall ┃ F1    ┃ Samples               ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ claude-sonnet   │ 0.81      │ 0.79   │ 0.80  │ 50                    │
│ gpt4o           │ 0.77      │ 0.74   │ 0.75  │ 50                    │
│ gemini-1.5-pro  │ 0.73      │ 0.70   │ 0.71  │ 50                    │
│ ollama:llama3   │ 0.61      │ 0.58   │ 0.59  │ 50                    │
└─────────────────┴───────────┴────────┴───────┴───────────────────────┘

Per-category breakdown (claude-sonnet):
  null_dereference  F1: 0.85
  sql_injection     F1: 0.90
  race_condition    F1: 0.74
  off_by_one        F1: 0.76
  resource_leak     F1: 0.77

Results saved → results/run_20240601_143022.json
```

---

## Project Structure

```
review-bench/
├── pyproject.toml              # Project metadata, dependencies, CLI entry-point
├── README.md
├── data/
│   └── samples.jsonl           # Curated benchmark dataset (50+ buggy code snippets)
├── review_bench/
│   ├── __init__.py             # Package marker and version constant
│   ├── cli.py                  # Typer CLI: run, list-samples, show-results
│   ├── models.py               # Adapters for Claude, GPT-4, Gemini, Ollama
│   ├── benchmark.py            # Orchestrates model runs against all samples
│   ├── scorer.py               # Computes precision, recall, F1 per model
│   ├── samples.py              # Loads and parses JSONL dataset into dataclasses
│   └── reporter.py             # Rich terminal table + JSON file export
└── tests/
    ├── conftest.py             # Shared fixtures and mock API responses
    ├── test_scorer.py          # Unit tests for precision/recall/F1 calculations
    ├── test_samples.py         # JSONL loader parsing and validation tests
    ├── test_models.py          # Model adapter tests with mocked HTTP responses
    ├── test_benchmark.py       # Benchmark orchestrator tests
    └── test_reporter.py        # Reporter rendering and JSON serialisation tests
```

---

## Configuration

review_bench is configured through environment variables and CLI flags. No config file is required.

### Environment Variables

| Variable | Description | Required for |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | `gpt4o` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `claude-opus`, `claude-sonnet` |
| `GOOGLE_API_KEY` | Google AI API key | `gemini` |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) | `ollama:*` |

### CLI Options — `review-bench run`

| Flag | Default | Description |
|---|---|---|
| `--models` | (required) | One or more model identifiers to benchmark |
| `--category` | all | Filter samples to a single bug category |
| `--language` | all | Filter samples to a single language |
| `--max-samples` | unlimited | Max samples per category (useful for quick runs) |
| `--output-dir` | `.` | Directory where the JSON results file is written |
| `--no-export` | false | Skip saving results to JSON |
| `--verbose` | false | Enable debug logging |

### Supported Model Identifiers

| Identifier | Provider | Model |
|---|---|---|
| `gpt4o` | OpenAI | gpt-4o |
| `claude-opus` | Anthropic | claude-3-opus-20240229 |
| `claude-sonnet` | Anthropic | claude-3-sonnet-20240229 |
| `gemini` | Google | gemini-1.5-pro |
| `ollama:<name>` | Local Ollama | Any model served by Ollama |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use mocked API responses via `respx` and `unittest.mock`—no real API keys are required.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
