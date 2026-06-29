"""YAML eval spec parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class SpecError(ValueError):
    """Raised when an eval spec is missing required fields."""


@dataclass
class DatasetSpec:
    path: Path
    format: str | None = None


@dataclass
class SplitSpec:
    ratios: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "test": 0.2})
    seed: int = 0
    target: str = "test"


@dataclass
class AppSpec:
    type: str
    ref: str | None = None
    url: str | None = None
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    working_dir: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMSpec:
    provider: str = "openai"
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorSpec:
    type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSpec:
    output_dir: Path = Path("reports")
    formats: list[str] = field(default_factory=lambda: ["json", "markdown"])
    run_name: str = "evalkit-run"


@dataclass
class PromptOptimizationSpec:
    enabled: bool = False
    app_param: str = "prompt"
    base_prompt: str | None = None
    candidates: list[str] = field(default_factory=list)
    num_variants: int = 4
    train_split: str = "train"
    eval_split: str = "test"


@dataclass
class EvalSpec:
    path: Path
    dataset: DatasetSpec
    app: AppSpec
    evaluators: list[EvaluatorSpec]
    split: SplitSpec = field(default_factory=SplitSpec)
    llm: LLMSpec | None = None
    report: ReportSpec = field(default_factory=ReportSpec)
    prompt_optimization: PromptOptimizationSpec = field(default_factory=PromptOptimizationSpec)


def load_spec(path: str | Path) -> EvalSpec:
    spec_path = Path(path).resolve()
    if not spec_path.exists():
        raise SpecError(f"Spec not found: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise SpecError("Eval spec must be a YAML mapping.")

    base_dir = spec_path.parent
    dataset = _parse_dataset(raw.get("dataset"), base_dir)
    app = _parse_app(raw.get("app"), base_dir)
    evaluators = _parse_evaluators(raw.get("evaluators"))
    split = _parse_split(raw.get("split"))
    llm = _parse_llm(raw.get("llm"))
    report = _parse_report(raw.get("report"), base_dir)
    prompt_optimization = _parse_prompt_optimization(raw.get("prompt_optimization"))

    return EvalSpec(
        path=spec_path,
        dataset=dataset,
        app=app,
        evaluators=evaluators,
        split=split,
        llm=llm,
        report=report,
        prompt_optimization=prompt_optimization,
    )


def _parse_dataset(raw: Any, base_dir: Path) -> DatasetSpec:
    if not isinstance(raw, dict):
        raise SpecError("Missing required 'dataset' mapping.")
    path_value = raw.get("path")
    if not path_value:
        raise SpecError("dataset.path is required.")
    return DatasetSpec(path=_resolve_path(base_dir, path_value), format=raw.get("format"))


def _parse_app(raw: Any, base_dir: Path) -> AppSpec:
    if not isinstance(raw, dict):
        raise SpecError("Missing required 'app' mapping.")
    app_type = str(raw.get("type") or "").lower()
    if not app_type:
        raise SpecError("app.type is required.")
    working_dir = raw.get("working_dir")
    return AppSpec(
        type=app_type,
        ref=raw.get("ref"),
        url=raw.get("url"),
        method=str(raw.get("method", "POST")).upper(),
        headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
        timeout_seconds=float(raw.get("timeout_seconds", raw.get("timeout", 30.0))),
        working_dir=_resolve_path(base_dir, working_dir) if working_dir else base_dir,
        params=dict(raw.get("params") or {}),
    )


def _parse_evaluators(raw: Any) -> list[EvaluatorSpec]:
    if not isinstance(raw, list) or not raw:
        raise SpecError("At least one evaluator is required.")
    evaluators: list[EvaluatorSpec] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SpecError(f"evaluators[{index}] must be a mapping.")
        eval_type = str(item.get("type") or "").lower()
        if not eval_type:
            raise SpecError(f"evaluators[{index}].type is required.")
        params = {key: value for key, value in item.items() if key not in {"type", "name"}}
        evaluators.append(
            EvaluatorSpec(
                type=eval_type,
                name=str(item.get("name") or eval_type),
                params=params,
            )
        )
    return evaluators


def _parse_split(raw: Any) -> SplitSpec:
    if raw is None:
        return SplitSpec()
    if not isinstance(raw, dict):
        raise SpecError("split must be a mapping.")

    ratios_raw = raw.get("ratios")
    if ratios_raw is None:
        ratios_raw = {
            key: value
            for key, value in raw.items()
            if key not in {"seed", "target"} and isinstance(value, int | float)
        }
    if not ratios_raw:
        ratios_raw = {"train": 0.8, "test": 0.2}
    ratios = {str(key): float(value) for key, value in dict(ratios_raw).items()}
    return SplitSpec(
        ratios=ratios,
        seed=int(raw.get("seed", 0)),
        target=str(raw.get("target", "test")),
    )


def _parse_llm(raw: Any) -> LLMSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SpecError("llm must be a mapping.")
    params = {key: value for key, value in raw.items() if key not in {"provider", "model"}}
    return LLMSpec(
        provider=str(raw.get("provider", "openai")).lower(),
        model=raw.get("model"),
        params=params,
    )


def _parse_report(raw: Any, base_dir: Path) -> ReportSpec:
    if raw is None:
        return ReportSpec(output_dir=base_dir / "reports")
    if not isinstance(raw, dict):
        raise SpecError("report must be a mapping.")
    formats = raw.get("formats", ["json", "markdown"])
    if isinstance(formats, str):
        formats = [formats]
    output_dir = raw.get("output_dir", "reports")
    return ReportSpec(
        output_dir=_resolve_path(base_dir, output_dir),
        formats=[str(item).lower() for item in formats],
        run_name=str(raw.get("run_name", "evalkit-run")),
    )


def _parse_prompt_optimization(raw: Any) -> PromptOptimizationSpec:
    if raw is None:
        return PromptOptimizationSpec()
    if not isinstance(raw, dict):
        raise SpecError("prompt_optimization must be a mapping.")
    candidates = raw.get("candidates", [])
    if isinstance(candidates, str):
        candidates = [candidates]
    return PromptOptimizationSpec(
        enabled=bool(raw.get("enabled", False)),
        app_param=str(raw.get("app_param", "prompt")),
        base_prompt=raw.get("base_prompt"),
        candidates=[str(item) for item in candidates],
        num_variants=int(raw.get("num_variants", 4)),
        train_split=str(raw.get("train_split", "train")),
        eval_split=str(raw.get("eval_split", "test")),
    )


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
