"""Adapters for invoking the app being evaluated."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .dataset import DataRow
from .spec import AppSpec


class AppError(RuntimeError):
    """Raised when the app under test cannot be invoked."""


@dataclass
class AppResult:
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AppUnderTest(Protocol):
    def run(self, row: DataRow) -> AppResult:
        """Run the app against one dataset row."""

    def with_params(self, overrides: dict[str, Any]) -> "AppUnderTest":
        """Return an equivalent app adapter with merged params."""


def build_app(spec: AppSpec) -> AppUnderTest:
    if spec.type == "callable":
        if not spec.ref:
            raise AppError("callable app requires app.ref, for example 'app:answer'.")
        return CallableApp(spec.ref, spec.working_dir or Path.cwd(), spec.params)
    if spec.type == "http":
        if not spec.url:
            raise AppError("http app requires app.url.")
        return HttpApp(
            url=spec.url,
            method=spec.method,
            headers=spec.headers,
            timeout_seconds=spec.timeout_seconds,
            params=spec.params,
        )
    raise AppError(f"Unsupported app.type '{spec.type}'. Use callable or http.")


@dataclass
class CallableApp:
    ref: str
    working_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    _callable: Callable[..., Any] | None = field(default=None, init=False, repr=False)

    def run(self, row: DataRow) -> AppResult:
        func = self._load()
        payload = {
            "input": row.input,
            "question": row.input,
            "expected_output": row.expected_output,
            "context": row.context,
            "metadata": row.metadata,
            **self.params,
        }
        result = _call_with_supported_kwargs(func, payload)
        return _coerce_app_result(result)

    def with_params(self, overrides: dict[str, Any]) -> "CallableApp":
        merged = {**self.params, **overrides}
        return CallableApp(self.ref, self.working_dir, merged)

    def _load(self) -> Callable[..., Any]:
        if self._callable is None:
            self._callable = _load_callable(self.ref, self.working_dir)
        return self._callable


@dataclass
class HttpApp:
    url: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    params: dict[str, Any] = field(default_factory=dict)

    def run(self, row: DataRow) -> AppResult:
        body = {
            "input": row.input,
            "question": row.input,
            "context": row.context,
            "metadata": row.metadata,
            **self.params,
        }
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        request = urllib.request.Request(self.url, data=data, method=self.method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise AppError(f"HTTP app call failed for {self.url}: {exc}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = response_body
        return _coerce_app_result(parsed)

    def with_params(self, overrides: dict[str, Any]) -> "HttpApp":
        return HttpApp(
            url=self.url,
            method=self.method,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
            params={**self.params, **overrides},
        )


def _load_callable(ref: str, working_dir: Path) -> Callable[..., Any]:
    if ":" not in ref:
        raise AppError("Callable ref must be in 'module:function' or 'path.py:function' form.")
    module_ref, func_name = ref.split(":", 1)
    if not func_name:
        raise AppError(f"Callable ref has no function name: {ref}")

    module_path = Path(module_ref)
    if not module_path.is_absolute():
        module_path = working_dir / module_path

    if module_ref.endswith(".py") or module_path.exists():
        if not module_path.exists():
            raise AppError(f"Callable module file not found: {module_path}")
        module_name = f"evalkit_app_{abs(hash(module_path))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise AppError(f"Could not load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        inserted = False
        working_dir_str = str(working_dir)
        if working_dir_str not in sys.path:
            sys.path.insert(0, working_dir_str)
            inserted = True
        try:
            module = importlib.import_module(module_ref)
        finally:
            if inserted:
                try:
                    sys.path.remove(working_dir_str)
                except ValueError:
                    pass

    func = getattr(module, func_name, None)
    if not callable(func):
        raise AppError(f"Callable '{func_name}' not found in {module_ref}.")
    return func


def _call_with_supported_kwargs(func: Callable[..., Any], payload: dict[str, Any]) -> Any:
    signature = inspect.signature(func)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(**payload)

    kwargs = {
        name: payload[name]
        for name, param in parameters.items()
        if name in payload
        and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return func(**kwargs)


def _coerce_app_result(result: Any) -> AppResult:
    if isinstance(result, AppResult):
        return result
    if isinstance(result, dict):
        output = result.get("output", result.get("answer", result.get("response", "")))
        metadata = {key: value for key, value in result.items() if key not in {"output", "answer", "response"}}
        return AppResult(output=str(output), metadata=metadata)
    return AppResult(output=str(result))
