# evalkit

> A reusable eval harness that ships *inside* AI accelerators — grade any AI app against a
> test dataset, find out **where** it's failing, and improve it by turning a small set of knobs.

**Status:** Early v1. The local CLI can load a dataset, run a pluggable app, score outputs,
cluster failures, write reports, and demonstrate a prompt-optimization loop.

Most eval tools give you a *score*. This accelerator closes the loop: it ties each score to a
**specific optimization action** (prompt, architecture, agent, or model) and re-measures the lift.
It's packaged so a delivery team can drop it onto an app and get *"here's what's wrong and here's
the knob to turn,"* not just a number.

## The loop

```
test dataset  ─►  run app  ─►  evaluators score outputs  ─►  cluster failures into failure modes
      ▲                                                                   │
      └──────────────  turn a "knob" (prompt / arch / agent / model)  ◄───┘
                              re-run → measure improvement
```

## The four knobs

| Knob | What you tune |
|------|---------------|
| **Prompt** | Wording, structure, templates ("generate N variants, keep the best") |
| **Architecture** | RAG pattern, chunking, retrieval |
| **Agent** | Tool use, multi-step flow |
| **Model** | Swap/compare models (provider-agnostic) |

## One thing per use case

Don't build a bespoke eval per app. Find the single capability that most improves it and grade that:

| Use case | The "one thing" to eval |
|----------|------------------------|
| Chatbot / RAG | Is the answer correct & grounded? |
| NL2SQL | NL → correct SQL |
| Entity extraction | Per-field accuracy |
| Agentic (many tools) | Tool use — right tool, right flow |

## v1 scope (chatbot / RAG)

1. Load a dataset of `(input, expected_output, [context])` rows (CSV / JSONL).
2. Run them through a pluggable **app under test**.
3. Score with pluggable **evaluators** (LLM-as-judge + deterministic).
4. Produce a **report** with per-item scores + **failure-mode clusters** (occurrence × severity × impact).
5. Train / test / validation splits, metrics on the held-out set.
6. **Prompt-optimizer demo:** generate N prompt variants, keep the best, show before/after lift.

## Quickstart

```bash
pip install -e .
evalkit run examples/rag_chatbot/spec.yaml
```

The bundled `examples/rag_chatbot` run uses deterministic scoring and does **not** require an API
key. LLM-as-judge evaluators do require OpenAI or Azure OpenAI configuration:

```bash
cp .env.example .env
# Fill OPENAI_API_KEY or AZURE_OPENAI_* values, then reference an llm block in your spec.
```

The run writes both JSON and Markdown reports:

```text
examples/rag_chatbot/reports/rag-chatbot-example.json
examples/rag_chatbot/reports/rag-chatbot-example.md
```

## Eval Spec

A spec defines the dataset, app under test, evaluators, split policy, reporting, and optional prompt
optimizer:

```yaml
dataset:
  path: dataset.jsonl
  format: jsonl

split:
  seed: 7
  target: test
  ratios:
    train: 0.5
    test: 0.5

app:
  type: callable
  ref: app:answer
  working_dir: .
  params:
    prompt: Answer the user helpfully.

evaluators:
  - type: exact_match
    name: exact_answer

prompt_optimization:
  enabled: true
  app_param: prompt
  base_prompt: Answer the user helpfully.
  candidates:
    - Answer the user using only the provided context.
```

Datasets can be CSV or JSONL with `input`, `expected_output`, and optional `context`. Common aliases
like `question` and `answer` are accepted.

Apps can be:

- `callable`: a local `module:function` or `path.py:function`.
- `http`: a JSON HTTP endpoint that receives `input`, `question`, `context`, `metadata`, and app
  params.

Evaluators currently include:

- `exact_match`
- `regex_match`
- `json_field_match`
- `llm_judge` (`correctness`, `relevance`, `groundedness`, or any named criterion)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

Built in milestones: scaffold, dataset/splits, app-under-test + LLM client, evaluators, report +
failure-mode clustering, prompt-optimizer knob, runnable example, docs polish.

**Definition of done for v1:** clone the repo, run the example with your own API key, and get a
report that says *"here's where your chatbot fails, clustered by failure mode, and here's the
prompt variant that improved it."*

## Contributing

Issues and PRs welcome. The full design spec and milestone plan are maintained by the core team.

## License

[MIT](LICENSE).
