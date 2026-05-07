# LLM Splitter Notebook Audit

Source notebook: `notebooks/Split_llm.ipynb`

---

## Model & API

- **Provider:** Hugging Face Hub (self-hosted inference via `transformers`)
- **Model (default):** `Qwen/Qwen2.5-7B-Instruct`
- **Alternatives mentioned in comments:** `ai-forever/ruGPT-3.5-13B`, `meta-llama/Llama-3.1-8B-Instruct`
- **Client library:** `transformers.pipeline` (text-generation), with `apply_chat_template` for chat formatting
- **No external API calls** (OpenAI, Anthropic, etc.); everything runs locally or in Colab

---

## I/O format

**Input:**
- Single sentence string per call (`split_sentence(sentence: str)`)
- Batch processing is available via `split_dataset(sentences: List[str])`, which loops sentence-by-sentence with `tqdm`
- Prompt is a two-message chat (system + user):
  - System: instructs the model to split a complex Russian sentence into grammatically complete simple parts, preserve original wording, and return strict JSON with no markdown or explanation
  - User: `Раздели на простые: "{sentence}"`

**Output:**
- Raw model generation is parsed with a regex that strips markdown fences and extracts the first valid JSON object
- Expected JSON schema: `{"simple_parts": ["часть 1", "часть 2", ...]}`
- Return type of `split_sentence`: `{"original": str, "simple_parts": List[str]}`
- Return type of `split_dataset`: `pd.DataFrame` with columns `original`, `simple_parts`, `status` (and optionally `error`)
- Full dataset result is serialised to JSON via `df.to_json(..., orient="records", force_ascii=False, indent=2)`

---

## Reusability for simple-sentence validation

**Verdict: usable with minor adaptation, but not AS-IS.**

The notebook can be repurposed for simple-sentence validation — feeding a candidate "simple" sentence and checking whether the LLM returns a single-element `simple_parts` list (confirming simplicity). The core `split_sentence` method already returns a list, so `len(result["simple_parts"]) == 1` is a ready-made simplicity check.

Blockers for immediate reuse:

1. **GPU / environment dependency** — `device="cuda"` is hard-coded in the demo cell (cell 1). On CPU or MPS (Apple Silicon) this must be changed.
2. **Model download size** — `Qwen2.5-7B-Instruct` is ~15 GB. First run requires bandwidth and disk space; no local cache path is configurable in the notebook.
3. **HF_TOKEN** — the notebook ran in Google Colab where `HF_TOKEN` is read from Colab Secrets. Locally this must be set as an environment variable (`export HF_TOKEN=...`) for authenticated Hub access (needed for gated models like Llama-3.1-8B-Instruct, optional for Qwen).
4. **Hard-coded input data path** — cell 2 reads `data.csv` from the current working directory with no path argument; this file is not present in the repo.
5. **Hard-coded output path** — `output_path="split_results.json"` and `"split_results_100.json"` write to cwd; these should be pointed at `notebooks/sampling/` or a data directory.
6. **Deprecated `torch_dtype` kwarg** — warning logged at runtime: use `dtype=` instead; harmless but worth cleaning up.
7. **`max_length=20` conflict** — the pipeline's default `max_length` conflicts with `max_new_tokens=256`; `max_new_tokens` wins, but the warning indicates the pipeline config should be cleaned up.
8. **No dependency manifest** — no `requirements.txt` or `pyproject.toml` pins `transformers`, `torch`, `pandas`, or `tqdm` versions.

---

## Required changes / blockers

| Priority | Change |
|----------|--------|
| Must | Replace `device="cuda"` with `device_map="auto"` or make it a parameter |
| Must | Accept input sentences from a variable/file rather than hard-coded `data.csv` |
| Must | Redirect output paths to repo-relative locations |
| Should | Set `HF_TOKEN` via env var or document the secret setup step |
| Should | Add `requirements.txt` with pinned versions |
| Nice-to-have | Fix deprecated `torch_dtype` → `dtype` and resolve `max_length` / `max_new_tokens` conflict |
| Nice-to-have | Add a thin wrapper `is_simple(sentence) -> bool` that calls `split_sentence` and checks `len(simple_parts) == 1` |
