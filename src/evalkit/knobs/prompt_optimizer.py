"""Prompt optimization knob demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from evalkit.app_under_test import AppUnderTest
from evalkit.dataset import DataRow
from evalkit.evaluators.base import Evaluator
from evalkit.llm_client import LLMClient


@dataclass
class PromptCandidateResult:
    prompt: str
    mean_score: float
    evaluator_scores: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "mean_score": round(self.mean_score, 6),
            "evaluator_scores": {key: round(value, 6) for key, value in self.evaluator_scores.items()},
        }


@dataclass
class PromptOptimizationResult:
    app_param: str
    train_split: str
    eval_split: str
    baseline_prompt: str
    best_prompt: str
    baseline_score: float
    best_score: float
    lift: float
    candidates: list[PromptCandidateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "app_param": self.app_param,
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "baseline_prompt": self.baseline_prompt,
            "best_prompt": self.best_prompt,
            "baseline_score": round(self.baseline_score, 6),
            "best_score": round(self.best_score, 6),
            "lift": round(self.lift, 6),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


EvaluationFunction = Callable[[list[DataRow], AppUnderTest, list[Evaluator], LLMClient | None, str], object]


class PromptOptimizer:
    def __init__(
        self,
        *,
        app_param: str,
        base_prompt: str,
        candidates: list[str],
        num_variants: int,
        train_split: str,
        eval_split: str,
    ) -> None:
        self.app_param = app_param
        self.base_prompt = base_prompt
        self.candidates = candidates
        self.num_variants = num_variants
        self.train_split = train_split
        self.eval_split = eval_split

    def optimize(
        self,
        *,
        splits: dict[str, list[DataRow]],
        app: AppUnderTest,
        evaluators: list[Evaluator],
        llm_client: LLMClient | None,
        evaluate_func: EvaluationFunction,
    ) -> PromptOptimizationResult:
        train_rows = splits.get(self.train_split)
        if train_rows is None:
            raise ValueError(f"Prompt optimizer train split '{self.train_split}' does not exist.")

        prompts = self._candidate_prompts()
        candidate_results: list[PromptCandidateResult] = []
        for prompt in prompts:
            candidate_app = app.with_params({self.app_param: prompt})
            run_result = evaluate_func(train_rows, candidate_app, evaluators, llm_client, self.train_split)
            candidate_results.append(
                PromptCandidateResult(
                    prompt=prompt,
                    mean_score=run_result.summary["mean_score"],
                    evaluator_scores={
                        name: metric["mean_score"] for name, metric in run_result.summary["evaluators"].items()
                    },
                )
            )

        baseline = candidate_results[0]
        best = max(candidate_results, key=lambda item: item.mean_score)
        return PromptOptimizationResult(
            app_param=self.app_param,
            train_split=self.train_split,
            eval_split=self.eval_split,
            baseline_prompt=self.base_prompt,
            best_prompt=best.prompt,
            baseline_score=baseline.mean_score,
            best_score=best.mean_score,
            lift=best.mean_score - baseline.mean_score,
            candidates=candidate_results,
        )

    def _candidate_prompts(self) -> list[str]:
        prompts = [self.base_prompt, *self.candidates]
        fallback = [
            self.base_prompt + "\nUse only the provided context. If the answer is present, answer exactly and concisely.",
            self.base_prompt + "\nGround every answer in the context and do not add unsupported details.",
            self.base_prompt + "\nExtract the relevant fact from context first, then answer in the expected wording.",
        ]
        prompts.extend(fallback)

        unique: list[str] = []
        for prompt in prompts:
            cleaned = prompt.strip()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
            if len(unique) >= max(1, self.num_variants):
                break
        return unique
