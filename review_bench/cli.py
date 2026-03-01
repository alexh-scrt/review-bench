"""Typer CLI entry-point for review_bench.

Provides three commands:

- ``run``          — Execute the benchmark suite against one or more model adapters.
- ``list-samples`` — Display all bundled benchmark samples, with optional filters.
- ``show-results`` — Render a previously saved results JSON file as a scorecard.

All terminal output uses Rich for colour and table formatting. Progress is
shown during ``run`` via a Rich live progress bar.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import box

from review_bench import __version__
from review_bench.benchmark import run_benchmark
from review_bench.models import (
    ADAPTER_REGISTRY,
    ClaudeAdapter,
    GeminiAdapter,
    GPT4Adapter,
    ModelAdapterError,
    OllamaAdapter,
    get_adapter,
    list_adapters,
)
from review_bench.reporter import (
    load_results,
    render_results_from_file,
    render_scorecard,
    save_results,
)
from review_bench.samples import (
    BenchmarkSample,
    SamplesLoadError,
    SampleValidationError,
    get_categories,
    get_languages,
    load_samples,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="review-bench",
    help="Evaluate AI-powered code review setups against a curated benchmark.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Shared Rich console for all commands.
_console = Console()
_err_console = Console(stderr=True, style="red")


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:  # noqa: FBT001
    if value:
        _console.print(f"review-bench {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Optional[bool] = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose debug logging.",
    ),
) -> None:
    """review_bench: objective AI code-review benchmarking."""
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )


# ---------------------------------------------------------------------------
# Helper: build adapters from CLI flags
# ---------------------------------------------------------------------------


def _build_adapters(
    model_ids: list[str],
    ollama_model: str,
    ollama_url: str,
) -> list:  # list[BaseReviewAdapter]
    """Instantiate model adapters from the given model ID list.

    Args:
        model_ids: List of adapter IDs selected by the user.
        ollama_model: Ollama model name to use when the ``ollama`` adapter is selected.
        ollama_url: Base URL for the Ollama server.

    Returns:
        List of instantiated adapter objects.

    Raises:
        typer.Exit: On unrecognised adapter IDs or import errors.
    """
    adapters = []
    known = set(list_adapters())

    for model_id in model_ids:
        if model_id not in known:
            _err_console.print(
                f"[bold red]Error:[/bold red] Unknown model '{model_id}'. "
                f"Choose from: {', '.join(sorted(known))}"
            )
            raise typer.Exit(code=1)

        try:
            if model_id == "ollama":
                adapter = OllamaAdapter(model=ollama_model, base_url=ollama_url)
            else:
                adapter = get_adapter(model_id)
        except ImportError as exc:
            _err_console.print(
                f"[bold red]Import error for '{model_id}':[/bold red] {exc}\n"
                "Make sure the required SDK is installed."
            )
            raise typer.Exit(code=1) from exc
        except Exception as exc:  # noqa: BLE001
            _err_console.print(
                f"[bold red]Failed to initialise adapter '{model_id}':[/bold red] {exc}"
            )
            raise typer.Exit(code=1) from exc

        adapters.append(adapter)

    return adapters


# ---------------------------------------------------------------------------
# Command: run
# ---------------------------------------------------------------------------


@app.command("run")
def cmd_run(
    model: Optional[list[str]] = typer.Option(  # noqa: UP007
        None,
        "--model",
        "-m",
        help=(
            "Model adapter ID(s) to evaluate. May be repeated. "
            f"Choices: {', '.join(sorted(ADAPTER_REGISTRY))}. "
            "Defaults to all available adapters."
        ),
        show_default=False,
    ),
    category: Optional[list[str]] = typer.Option(  # noqa: UP007
        None,
        "--category",
        "-c",
        help=(
            "Only include samples from this category. May be repeated. "
            "E.g. --category sql_injection --category null_dereference"
        ),
        show_default=False,
    ),
    language: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--language",
        "-l",
        help="Only include samples written in this language (e.g. python, java).",
    ),
    max_samples: Optional[int] = typer.Option(  # noqa: UP007
        None,
        "--max-samples",
        "-n",
        help="Maximum number of samples to evaluate per category.",
        min=1,
    ),
    output: Optional[Path] = typer.Option(  # noqa: UP007
        None,
        "--output",
        "-o",
        help="Path for the output JSON results file.",
        show_default=False,
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        help="Directory in which to save timestamped results (ignored if --output is set).",
        show_default=True,
    ),
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        help="Maximum concurrent API requests per adapter.",
        min=1,
        max=20,
    ),
    ollama_model: str = typer.Option(
        "llama3",
        "--ollama-model",
        help="Ollama model name (used when --model ollama is selected).",
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434",
        "--ollama-url",
        help="Ollama server base URL.",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Do not save results to disk; only display the scorecard.",
    ),
    no_category_breakdown: bool = typer.Option(
        False,
        "--no-category-breakdown",
        help="Skip per-category breakdown in the scorecard.",
    ),
    samples_file: Optional[Path] = typer.Option(  # noqa: UP007
        None,
        "--samples-file",
        help="Path to a custom JSONL samples file (defaults to bundled dataset).",
    ),
) -> None:
    """[bold]Run[/bold] the benchmark against one or more model adapters.

    Examples:

    \b
      review-bench run
      review-bench run --model gpt4 --model claude
      review-bench run --category sql_injection --max-samples 5
      review-bench run --model ollama --ollama-model codellama
    """
    # Resolve model list.
    model_ids: list[str] = list(model) if model else list(sorted(ADAPTER_REGISTRY.keys()))

    # Validate category filter early.
    if category:
        try:
            all_samples = load_samples(
                path=str(samples_file) if samples_file else None,
                skip_invalid=True,
            )
        except SamplesLoadError as exc:
            _err_console.print(f"[bold red]Cannot load samples:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

        valid_categories = get_categories(all_samples)
        for cat in category:
            if cat not in valid_categories:
                _err_console.print(
                    f"[bold red]Error:[/bold red] Unknown category '{cat}'. "
                    f"Valid categories: {', '.join(valid_categories)}"
                )
                raise typer.Exit(code=1)

    # Build adapters.
    adapters = _build_adapters(model_ids, ollama_model, ollama_url)

    _console.print()
    _console.print(
        Panel(
            f"[bold]Models:[/bold] {', '.join(a.model_name for a in adapters)}\n"
            f"[bold]Category filter:[/bold] {', '.join(category) if category else 'all'}\n"
            f"[bold]Language filter:[/bold] {language or 'all'}\n"
            f"[bold]Max samples/category:[/bold] {max_samples or 'unlimited'}\n"
            f"[bold]Concurrency:[/bold] {concurrency}",
            title="[bold cyan]review_bench — Starting Run[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
    _console.print()

    # Set up Rich progress tracking.
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    )

    # Track per-adapter task IDs.
    adapter_tasks: dict[str, TaskID] = {}
    for adapter in adapters:
        task_id = progress.add_task(
            f"[cyan]{adapter.model_name}[/cyan]",
            total=None,  # Unknown until samples loaded
        )
        adapter_tasks[adapter.adapter_id] = task_id

    # Build progress callback.
    def _progress_callback(
        adapter_id: str, sample_id: str, current: int, total: int
    ) -> None:
        task_id = adapter_tasks.get(adapter_id)
        if task_id is not None:
            progress.update(
                task_id,
                completed=current,
                total=total,
                description=f"[cyan]{adapter_id}[/cyan] ({sample_id})",
            )

    # Run benchmark.
    benchmark_run = None
    try:
        with progress:
            # Resolve category: benchmark supports single category filter at a time;
            # we run multiple categories by not filtering (or running per-cat).
            # For simplicity, pass the first category if given; otherwise None.
            # The samples module handles filtering before passing to orchestrator.
            # We directly load + filter samples here for multi-category support.

            # Build multi-category filter: load samples matching ANY of the categories.
            if category and len(category) > 1:
                from review_bench.samples import load_samples as _load

                filtered_samples: list[BenchmarkSample] = []
                seen_ids: set[str] = set()
                for cat in category:
                    for s in _load(
                        path=str(samples_file) if samples_file else None,
                        category=cat,
                        language=language,
                        skip_invalid=True,
                    ):
                        if s.id not in seen_ids:
                            filtered_samples.append(s)
                            seen_ids.add(s.id)

                # Temporarily patch load_samples to return our filtered list.
                import review_bench.samples as _samples_module
                import unittest.mock as _mock

                with _mock.patch.object(
                    _samples_module, "load_samples", return_value=filtered_samples
                ):
                    benchmark_run = asyncio.run(
                        run_benchmark(
                            adapters,
                            max_samples=max_samples,
                            concurrency=concurrency,
                            progress_callback=_progress_callback,
                            skip_invalid_samples=True,
                        )
                    )
            else:
                benchmark_run = asyncio.run(
                    run_benchmark(
                        adapters,
                        samples_path=str(samples_file) if samples_file else None,
                        category_filter=category[0] if category else None,
                        language_filter=language,
                        max_samples=max_samples,
                        concurrency=concurrency,
                        progress_callback=_progress_callback,
                        skip_invalid_samples=True,
                    )
                )
    except SamplesLoadError as exc:
        _err_console.print(f"[bold red]Failed to load samples:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        _err_console.print("\n[yellow]Benchmark interrupted by user.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as exc:  # noqa: BLE001
        _err_console.print(f"[bold red]Benchmark failed:[/bold red] {exc}")
        logger.debug("Benchmark error detail:", exc_info=True)
        raise typer.Exit(code=1) from exc

    if benchmark_run is None:
        _err_console.print("[bold red]Benchmark produced no results.[/bold red]")
        raise typer.Exit(code=1)

    # Render scorecard.
    render_scorecard(
        benchmark_run,
        console=_console,
        show_category_breakdown=not no_category_breakdown,
    )

    # Save results.
    if not no_save:
        try:
            if output:
                saved_path = save_results(benchmark_run, output_path=output)
            else:
                saved_path = save_results(benchmark_run, output_dir=output_dir)
            _console.print(
                f"[green]✓[/green] Results saved to [bold]{saved_path}[/bold]"
            )
        except OSError as exc:
            _err_console.print(f"[bold red]Failed to save results:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Command: list-samples
# ---------------------------------------------------------------------------


@app.command("list-samples")
def cmd_list_samples(
    category: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--category",
        "-c",
        help="Filter samples by category (e.g. null_dereference, sql_injection).",
    ),
    language: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--language",
        "-l",
        help="Filter samples by programming language (e.g. python, java).",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Include the code snippet in the output.",
    ),
    samples_file: Optional[Path] = typer.Option(  # noqa: UP007
        None,
        "--samples-file",
        help="Path to a custom JSONL samples file.",
    ),
) -> None:
    """[bold]List[/bold] available benchmark samples, with optional filtering.

    Examples:

    \b
      review-bench list-samples
      review-bench list-samples --category sql_injection
      review-bench list-samples --language python --show-code
    """
    try:
        samples = load_samples(
            path=str(samples_file) if samples_file else None,
            category=category,
            language=language,
            skip_invalid=True,
        )
    except SamplesLoadError as exc:
        _err_console.print(f"[bold red]Cannot load samples:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not samples:
        _console.print(
            "[yellow]No samples found matching the given filters.[/yellow]"
        )
        raise typer.Exit()

    # Summary line.
    cats = get_categories(samples)
    langs = get_languages(samples)
    _console.print()
    _console.print(
        Panel(
            f"[bold]Total samples:[/bold] {len(samples)}   "
            f"[bold]Categories:[/bold] {', '.join(cats)}   "
            f"[bold]Languages:[/bold] {', '.join(langs)}",
            title="[bold cyan]review_bench — Benchmark Samples[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
    _console.print()

    # Build table.
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        show_lines=show_code,
        expand=False,
        min_width=80,
    )
    table.add_column("ID", style="dim", min_width=12, no_wrap=True)
    table.add_column("Language", min_width=10, no_wrap=True)
    table.add_column("Category", min_width=18, no_wrap=True)
    table.add_column("Description", min_width=40)
    if show_code:
        table.add_column("Code", min_width=40)

    _CATEGORY_COLOURS: dict[str, str] = {
        "null_dereference": "red",
        "sql_injection": "magenta",
        "off_by_one": "yellow",
        "race_condition": "blue",
        "resource_leak": "green",
    }

    for sample in samples:
        colour = _CATEGORY_COLOURS.get(sample.category, "white")
        row: list[str] = [
            sample.id,
            sample.language,
            f"[{colour}]{sample.category}[/{colour}]",
            sample.description,
        ]
        if show_code:
            code_display = sample.code.strip()
            # Limit display length.
            if len(code_display) > 300:
                code_display = code_display[:297] + "..."
            row.append(f"[dim]{code_display}[/dim]")
        table.add_row(*row)

    _console.print(table)
    _console.print(
        f"[dim]{len(samples)} sample(s) displayed.[/dim]"
    )


# ---------------------------------------------------------------------------
# Command: show-results
# ---------------------------------------------------------------------------


@app.command("show-results")
def cmd_show_results(
    file: Optional[Path] = typer.Option(  # noqa: UP007
        None,
        "--file",
        "-f",
        help="Path to a results JSON file. If omitted, the most recent file in --results-dir is used.",
        show_default=False,
    ),
    results_dir: Path = typer.Option(
        Path("results"),
        "--results-dir",
        help="Directory to search for the most recent results file.",
        show_default=True,
    ),
    no_category_breakdown: bool = typer.Option(
        False,
        "--no-category-breakdown",
        help="Skip per-category breakdown tables.",
    ),
) -> None:
    """[bold]Show[/bold] a previously saved benchmark results scorecard.

    Examples:

    \b
      review-bench show-results
      review-bench show-results --file results/run_20240101_120000.json
      review-bench show-results --results-dir ./my_results
    """
    results_path: Path

    if file is not None:
        results_path = file
        if not results_path.exists():
            _err_console.print(
                f"[bold red]File not found:[/bold red] {results_path}"
            )
            raise typer.Exit(code=1)
    else:
        # Auto-discover the most recent JSON file in results_dir.
        if not results_dir.exists():
            _err_console.print(
                f"[bold red]Results directory does not exist:[/bold red] {results_dir}\n"
                "Run 'review-bench run' first to generate results."
            )
            raise typer.Exit(code=1)

        json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not json_files:
            _err_console.print(
                f"[bold red]No JSON result files found in:[/bold red] {results_dir}"
            )
            raise typer.Exit(code=1)

        results_path = json_files[-1]  # Most recent.
        _console.print(
            f"[dim]Loading most recent results: [bold]{results_path}[/bold][/dim]"
        )

    try:
        render_results_from_file(
            results_path,
            console=_console,
            show_category_breakdown=not no_category_breakdown,
        )
    except OSError as exc:
        _err_console.print(f"[bold red]Cannot read results file:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        _err_console.print(
            f"[bold red]Invalid results file format:[/bold red] {exc}"
        )
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Command: list-models
# ---------------------------------------------------------------------------


@app.command("list-models")
def cmd_list_models() -> None:
    """[bold]List[/bold] all available model adapters and their details."""
    _console.print()
    table = Table(
        title="[bold cyan]Available Model Adapters[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        expand=False,
    )
    table.add_column("Adapter ID", style="bold", min_width=14, no_wrap=True)
    table.add_column("Provider", min_width=12)
    table.add_column("Default Model", min_width=28)
    table.add_column("Env Var / Notes", min_width=30)

    _MODEL_INFO: list[tuple[str, str, str, str]] = [
        ("gpt4", "OpenAI", "gpt-4o", "OPENAI_API_KEY"),
        ("claude", "Anthropic", "claude-3-opus-20240229", "ANTHROPIC_API_KEY"),
        ("gemini", "Google", "gemini-1.5-pro", "GOOGLE_API_KEY"),
        ("ollama", "Local", "llama3 (configurable)", "No key — run: ollama serve"),
    ]

    for adapter_id, provider, default_model, notes in _MODEL_INFO:
        table.add_row(adapter_id, provider, default_model, f"[dim]{notes}[/dim]")

    _console.print(table)
    _console.print(
        "\n[dim]Use [bold]--model <adapter_id>[/bold] with the [bold]run[/bold] command "
        "to select specific adapters.[/dim]"
    )


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app()
