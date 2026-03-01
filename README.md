# review_bench

**review_bench** is a developer tool that objectively evaluates AI-powered code review setups—supporting Claude, GPT-4, Gemini, and local Ollama models—against a curated benchmark of buggy code samples. It produces precision/recall/F1 scorecards so your team can compare models without relying on vendor-provided benchmarks.

## Features

- **Multi-model support**: Claude (claude-3-opus/sonnet), GPT-4o, Gemini 1.5 Pro, and any Ollama-served local model
- **Standardized benchmark dataset**: 50 buggy code samples across five categories:
  - `null_dereference` — missing null/None checks
  - `off_by_one` — boundary and index errors
  - `sql_injection` — unsanitized query construction
  - `race_condition` — thread/async safety issues
  - `resource_leak` — unclosed files, connections, sockets
- **Precision/recall/F1 scoring**: keyword and category matching against ground-truth labels
- **Side-by-side comparison table**: Rich-powered terminal table ranked by F1 score with per-category breakdowns
- **Exportable results**: timestamped JSON output for CI integration or historical tracking

## Installation

```bash
# From PyPI (once published)
pip install review-bench

# From source
git clone https://github.com/example/review_bench
cd review_bench
pip install -e .[dev]
```

## Quick Start

### Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
# For Ollama: no key needed, just run `ollama serve`
```

### Run the Benchmark

```bash
# Run all configured models against the full benchmark
review-bench run

# Run only specific models
review-bench run --model gpt4 --model claude

# Run a subset of categories
review-bench run --category sql_injection --category null_dereference

# Limit number of samples per category
review-bench run --max-samples 5

# Save results to a custom path
review-bench run --output results/my_run.json
```

### List Available Samples

```bash
# Show all benchmark samples
review-bench list-samples

# Filter by category
review-bench list-samples --category race_condition

# Filter by language
review-bench list-samples --language python
```

### View Past Results

```bash
# Show the most recent results file
review-bench show-results

# Show a specific results file
review-bench show-results --file results/run_20240101_120000.json
```

## Example Scorecard Output

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    review_bench Scorecard — 2024-01-15                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Model             │ Precision │  Recall  │    F1    │ Samples  │  Time  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  gpt-4o            │   0.847   │  0.812   │  0.829   │   50     │  142s  ║
║  claude-3-opus     │   0.831   │  0.798   │  0.814   │   50     │  118s  ║
║  gemini-1.5-pro    │   0.803   │  0.776   │  0.789   │   50     │  201s  ║
║  llama3 (ollama)   │   0.721   │  0.694   │  0.707   │   50     │   89s  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                        Category Breakdown — gpt-4o                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Category          │ Precision │  Recall  │    F1    │ Samples           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  null_dereference  │   0.891   │  0.850   │  0.870   │   13              ║
║  sql_injection     │   0.923   │  0.900   │  0.911   │   11              ║
║  off_by_one        │   0.812   │  0.789   │  0.800   │   10              ║
║  resource_leak     │   0.845   │  0.810   │  0.827   │   10              ║
║  race_condition    │   0.778   │  0.733   │  0.755   │    9              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## Supported Models

| Model ID       | Provider    | Model Name            | Notes                        |
|----------------|-------------|----------------------|------------------------------|
| `gpt4`         | OpenAI      | gpt-4o               | Requires `OPENAI_API_KEY`    |
| `claude`       | Anthropic   | claude-3-opus-20240229 | Requires `ANTHROPIC_API_KEY` |
| `claude-sonnet`| Anthropic   | claude-3-5-sonnet-20240620 | Requires `ANTHROPIC_API_KEY` |
| `gemini`       | Google      | gemini-1.5-pro       | Requires `GOOGLE_API_KEY`    |
| `ollama`       | Local       | llama3 (configurable) | Requires running Ollama      |

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

### Adding Custom Samples

Append records to `data/samples.jsonl` following the schema above. Custom samples will be automatically included in benchmark runs.

## Scoring Methodology

review_bench uses **keyword matching** against the ground-truth `bug_labels` and `expected_issues` fields:

1. The model's review text is split into normalized tokens
2. Each expected issue keyword is checked for presence in the model's response
3. **Precision** = fraction of model-identified issues that are correct
4. **Recall** = fraction of expected issues identified by the model
5. **F1** = harmonic mean of precision and recall

Per-category scores are computed by grouping samples by the `category` field.

## Configuration

Create a `review_bench.toml` in your project root for persistent settings:

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

## Development

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Run tests with coverage
pytest --cov=review_bench
```

## License

MIT License — see [LICENSE](LICENSE) for details.
