# review_bench

**review_bench** is a developer tool that objectively evaluates AI-powered code review setups—supporting Claude, GPT-4, Gemini, and local Ollama models—against a curated benchmark of buggy code samples. It produces precision/recall/F1 scorecards so your team can compare models without relying on vendor-provided benchmarks.

## Features

- **Multi-model support**: Claude (claude-3-opus/sonnet), GPT-4o, Gemini 1.5 Pro, and any Ollama-served local model
- **Standardised benchmark dataset**: 50 buggy code samples across five categories:
  - `null_dereference` — missing null/None checks
  - `off_by_one` — boundary and index errors
  - `sql_injection` — unsanitised query construction
  - `race_condition` — thread/async safety issues
  - `resource_leak` — unclosed files, connections, sockets
- **Precision/recall/F1 scoring**: keyword and category matching against ground-truth labels
- **Side-by-side comparison table**: Rich-powered terminal table ranked by F1 score with per-category breakdowns
- **Exportable results**: timestamped JSON output for CI integration or historical tracking

## Installation

```bash
# From source
git clone https://github.com/example/review_bench
cd review_bench
pip install -e .[dev]
```

> **Python 3.11+ is required.**

## Quick Start

### 1. Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
# For Ollama: no key needed — just run `ollama serve`
```

### 2. Run the Benchmark

```bash
# Run all configured models against the full benchmark (50 samples)
review-bench run

# Run only specific models
review-bench run --model gpt4 --model claude

# Run a subset of categories
review-bench run --category sql_injection --category null_dereference

# Limit samples per category (useful for quick smoke tests)
review-bench run --max-samples 3

# Save results to a specific file
review-bench run --output my_results/run.json

# Run with a local Ollama model
review-bench run --model ollama --ollama-model codellama

# Run without saving to disk
review-bench run --no-save
```

### 3. List Available Samples

```bash
# Show all 50 benchmark samples
review-bench list-samples

# Filter by bug category
review-bench list-samples --category race_condition

# Filter by programming language
review-bench list-samples --language python

# Show code snippets inline
review-bench list-samples --category sql_injection --show-code
```

### 4. View Past Results

```bash
# Show the most recent results file in ./results/
review-bench show-results

# Show a specific results file
review-bench show-results --file results/run_20240115_120000.json

# Show results from a custom directory
review-bench show-results --results-dir ./my_results
```

### 5. List Available Models

```bash
review-bench list-models
```

## Example Scorecard Output

```
╭─────────────────────────────────────────────────────────────╮
│           review_bench Results                              │
│  Run ID: 20240115_120000   Samples: 50   Timestamp: ...     │
╰─────────────────────────────────────────────────────────────╯

╔══════════════════════════════════════════════════════════════════════════╗
║                    review_bench Scorecard                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Model                  │ Precision │  Recall  │    F1    │ Samples │ Time ║
╠══════════════════════════════════════════════════════════════════════════╣
║  gpt-4o (best)          │   0.847   │  0.812   │  0.829   │   50    │ 142s ║
║  claude-3-opus-20240229 │   0.831   │  0.798   │  0.814   │   50    │ 118s ║
║  gemini-1.5-pro         │   0.803   │  0.776   │  0.789   │   50    │ 201s ║
║  llama3                 │   0.721   │  0.694   │  0.707   │   50    │  89s ║
╚══════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━ Category Breakdown — gpt-4o ━━━━━━━━━━━━━━━━
 Category          Precision  Recall    F1      Samples
 sql_injection       0.923     0.900    0.911      11
 null_dereference    0.891     0.850    0.870      13
 resource_leak       0.845     0.810    0.827      10
 off_by_one          0.812     0.789    0.800      10
 race_condition      0.778     0.733    0.755       6
```

## CLI Reference

### `review-bench run`

Run the benchmark against one or more model adapters.

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | all adapters | Model adapter ID(s). Repeatable. |
| `--category, -c` | all | Only include samples from this category. Repeatable. |
| `--language, -l` | all | Only include samples in this language. |
| `--max-samples, -n` | unlimited | Max samples per category. |
| `--output, -o` | auto | Explicit output JSON path. |
| `--output-dir` | `results/` | Directory for timestamped output files. |
| `--concurrency` | `5` | Max concurrent API requests per adapter. |
| `--ollama-model` | `llama3` | Ollama model name. |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL. |
| `--no-save` | `False` | Skip saving results to disk. |
| `--no-category-breakdown` | `False` | Skip per-category table in scorecard. |
| `--samples-file` | bundled | Path to a custom JSONL samples file. |

### `review-bench list-samples`

Display benchmark samples with optional filtering.

| Flag | Default | Description |
|------|---------|-------------|
| `--category, -c` | all | Filter by bug category. |
| `--language, -l` | all | Filter by programming language. |
| `--show-code` | `False` | Include code snippets in output. |
| `--samples-file` | bundled | Path to a custom JSONL samples file. |

### `review-bench show-results`

Render a saved results JSON file as a scorecard.

| Flag | Default | Description |
|------|---------|-------------|
| `--file, -f` | most recent | Path to a specific results JSON file. |
| `--results-dir` | `results/` | Directory to search for result files. |
| `--no-category-breakdown` | `False` | Skip per-category tables. |

### `review-bench list-models`

List all available model adapters and their configuration.

### Global Flags

| Flag | Description |
|------|-------------|
| `--version, -V` | Show version and exit. |
| `--verbose, -v` | Enable debug logging. |

## Supported Models

| Adapter ID | Provider | Default Model | Required |
|------------|----------|--------------|----------|
| `gpt4` | OpenAI | `gpt-4o` | `OPENAI_API_KEY` |
| `claude` | Anthropic | `claude-3-opus-20240229` | `ANTHROPIC_API_KEY` |
| `gemini` | Google | `gemini-1.5-pro` | `GOOGLE_API_KEY` |
| `ollama` | Local | `llama3` (configurable) | Ollama server running |

## Benchmark Dataset

The benchmark contains 50 hand-curated buggy code samples stored in `data/samples.jsonl`. Each sample includes:

```json
{
  "id": "sample_001",
  "language": "python",
  "category": "null_dereference",
  "description": "Function dereferences a potentially None return value without checking.",
  "code": "def get_user_name(user_id):\n    user = find_user(user_id)\n    return user.name\n",
  "bug_labels": ["null_dereference", "missing_null_check", "attribute_error"],
  "expected_issues": ["null dereference", "none check", "missing null check", "user could be None"]
}
```

### Dataset Breakdown

| Category | Count | Languages |
|----------|-------|-----------|
| `null_dereference` | 13 | Python, JavaScript, Java, Go |
| `sql_injection` | 10 | Python, JavaScript, Java, Go |
| `resource_leak` | 10 | Python, JavaScript, Java, Go |
| `off_by_one` | 10 | Python, JavaScript, Java |
| `race_condition` | 7 | Python, JavaScript, Java, Go |

### Adding Custom Samples

Append records to `data/samples.jsonl` (or point to a custom file with `--samples-file`) following the schema above. New samples are automatically included in benchmark runs.

## Scoring Methodology

review_bench uses **keyword matching** against the ground-truth `bug_labels` and `expected_issues` fields:

1. The model's review text is split into normalised tokens (lowercased, punctuation removed).
2. Each `expected_issues` phrase is checked for presence in the model's response.
3. Matched issues are verified against `bug_labels` using bidirectional substring matching.
4. **Precision** = true positives / (true positives + false positives)
5. **Recall** = true positives / total expected issues
6. **F1** = harmonic mean of precision and recall
7. Scores are **micro-averaged** across all samples for aggregate metrics.

Per-category scores group samples by their `category` field before averaging.

## Results JSON Format

Results are saved as structured JSON for easy programmatic consumption:

```json
{
  "run_id": "20240115_120000",
  "timestamp": "2024-01-15T12:00:00+00:00",
  "total_samples": 50,
  "model_scores": [
    {
      "model_id": "gpt4",
      "model_name": "gpt-4o",
      "precision": 0.847,
      "recall": 0.812,
      "f1": 0.829,
      "sample_count": 50,
      "elapsed_seconds": 142.3,
      "category_scores": [
        {
          "category": "null_dereference",
          "precision": 0.891,
          "recall": 0.850,
          "f1": 0.870,
          "sample_count": 13
        }
      ]
    }
  ],
  "sample_results": [
    {
      "sample_id": "sample_001",
      "model_id": "gpt4",
      "identified_issues": ["null dereference", "none check"],
      "precision": 1.0,
      "recall": 0.5,
      "f1": 0.667,
      "raw_response": "..."
    }
  ]
}
```

## CI Integration

Use the exit code and JSON output for CI pipelines:

```bash
# Run benchmark and fail CI if the command errors
review-bench run --model gpt4 --max-samples 5 --output results/ci_run.json

# Check the F1 score programmatically
python -c "
import json, sys
data = json.load(open('results/ci_run.json'))
f1 = data['model_scores'][0]['f1']
print(f'F1 score: {f1:.3f}')
sys.exit(0 if f1 >= 0.7 else 1)
"
```

## Configuration (Optional)

Create a `review_bench.toml` in your project root for persistent defaults:

```toml
[models]
default = ["gpt4", "claude"]

[ollama]
base_url = "http://localhost:11434"
model = "llama3"

[benchmark]
max_samples = 50
output_dir = "results"
```

> **Note:** TOML config file support is not yet implemented. Use CLI flags.

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run the test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_scorer.py -v

# Run tests with coverage
pytest --cov=review_bench --cov-report=term-missing
```

### Project Structure

```
review_bench/
├── __init__.py        # Package version
├── cli.py             # Typer CLI commands
├── models.py          # Model adapter classes (GPT-4, Claude, Gemini, Ollama)
├── benchmark.py       # Benchmark orchestrator
├── scorer.py          # Precision/recall/F1 scorer
├── samples.py         # JSONL sample loader and dataclasses
└── reporter.py        # Rich terminal renderer and JSON exporter

data/
└── samples.jsonl      # 50 curated buggy code samples

tests/
├── conftest.py        # Shared pytest fixtures
├── test_scorer.py     # Scorer unit tests
├── test_samples.py    # Sample loader tests
├── test_models.py     # Model adapter tests (mocked HTTP)
├── test_benchmark.py  # Orchestrator tests
└── test_reporter.py   # Reporter and serialisation tests
```

## License

MIT License — see [LICENSE](LICENSE) for details.
