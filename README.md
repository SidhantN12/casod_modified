# CoT Distillation into Llama-3.2-1B

A lightweight, single-GPU reimplementation of the CasCoD pipeline: generate chain-of-thought data from a large teacher model, LoRA-distill it into a small student, and evaluate on BBH and out-of-domain reasoning benchmarks.

Derived from [C-W-D/CasCoD](https://github.com/C-W-D/CasCoD) (Dai et al., EMNLP 2024 — [paper](https://arxiv.org/abs/2405.19842)). The method design, benchmark selection, prompts, and evaluation harness originate there.

> **Scope note.** This repo currently implements **standard SFT distillation**, not CasCoD's cascading decomposed loss. CasCoD's contribution is a two-pass objective — `alpha * answer_loss + (1 - alpha) * rationale_loss` — that trains the student on rationales with the answer removed, then on the answer conditioned on that rationale. `training/lora_train.py` instead concatenates question and answer into a single sequence and trains one causal-LM loss over the whole thing. That corresponds to the Std-CoT baseline in the paper, not to CasCoD. Implementing the cascade is the main open item; see [Roadmap](#roadmap).

---

## Why this exists

The official implementation targets a Llama-2-7B student on multi-GPU FSDP with vLLM inference. This version rebuilds the same pipeline around **a single GPU with no distributed code**, and makes evaluation runnable on native Windows where vLLM is unavailable: a 1B student, plain Hugging Face `Trainer` + `peft`, and an optional-vLLM evaluation path. It also regenerates the teacher CoT data with a current OpenAI model rather than reusing the shipped 2023-era dump.

---

## What differs from upstream

| Component | Upstream CasCoD | This repo |
|---|---|---|
| **Training objective** | Two-pass cascading loss (`LLMCMTTool`, `alpha=0.1`) | Single causal-LM loss over `"Problem:\n{q}\n\nSolution:\n{a}"` |
| **Student model** | Llama-2-7B | Llama-3.2-1B (~7× smaller) |
| **Training infra** | FSDP + `torchrun`, multi-GPU | Single-GPU HF `Trainer` + `peft` |
| **Teacher model** | `gpt-4-1106-preview`, legacy `openai` 0.x API | `gpt-4.1`, `openai` ≥1.x client with legacy fallback |
| **Generation strategy** | Escalating retry loop, raising `n` and temperature on each failed attempt until the answer marker appears | Single fixed-temperature call requesting `n` completions, taking the first one carrying the marker |
| **Generation CLI** | One script per benchmark (`call_api_repair_bbh.py`, etc.) | Unified `--dataset bbh_train\|bbh_test\|bb_sub_test` |
| **LoRA config** | `r=64`, `alpha=32`, on `q_proj` and `v_proj` | `r=16`, `alpha=16`, on all seven linear projections (`q/k/v/o/gate/up/down`) |
| **Schedule** | 15 epochs, batch 16 × accum 4, `max_words=1024`, `weight_decay=0.05` | 3 epochs, batch 2 × accum 32, `max_len=768` |
| **Checkpointing** | FSDP conversion (`checkpoint_converter_fsdp_hf.py`, `batch_convert_peft.py`) | Per-epoch `SaveEpochCallback` writing adapters directly |
| **Evaluation** | vLLM required | vLLM optional (`_VLLM_AVAILABLE`), falls back to `transformers.generate()` |
| **Interactive inference** | Not present | `inference/run_lora_inference.py`, new in this repo |

Effective batch size matches upstream at 64, so the schedule difference is in epoch count and sequence length rather than batch statistics.

### Training data

Teacher CoTs were regenerated rather than reused, then filtered to keep only clean, correctly-answered traces:

| | Upstream `all_task_train_right_answer.json` | This repo `bbh_train_gpt41_strict_good.json` |
|---|---|---|
| Rows | 3,805 | 2,104 |
| BBH tasks covered | 27 | 17 |

The stricter filter combined with a different teacher produces a smaller and narrower training set. Ten BBH tasks end up with no training examples, so part of the nominally in-domain BBH evaluation is effectively out-of-domain for this student. That is worth holding in mind when comparing IND and OOD numbers.

---

## Repository layout

```
scripts/generate_cot.py          Teacher CoT generation (gpt-4.1), unified CLI
data/generated_cots/             Regenerated and filtered CoT training data
training/lora_train.py           LoRA SFT training loop, per-epoch adapter saving
evaluation/evaluation.py         Benchmark evaluation, vLLM-optional
inference/run_lora_inference.py  Interactive CLI for querying a trained adapter
adapters/epoch-3/                Trained LoRA adapter checkpoint
```

---

## Setup

```bash
git clone https://github.com/SidhantN12/casod_modified.git
cd casod_modified

conda create -n cascod python=3.10 -y
conda activate cascod

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets bitsandbytes openai
```

Llama-3.2-1B is gated on the Hugging Face Hub. Request access, then authenticate:

```bash
huggingface-cli login
```

Set an API key before regenerating data:

```bash
export OPENAI_API_KEY=...          # PowerShell: $env:OPENAI_API_KEY="..."
```

Training was run on a single rented NVIDIA L40S (RunPod). There is no distributed or multi-GPU code path, and the default batch size and sequence length are small enough that the run also fits on much smaller cards. vLLM is not required and is skipped automatically on Windows.

---

## Usage

### 1. Generate teacher CoTs

```bash
python scripts/generate_cot.py --dataset bbh_train
python scripts/generate_cot.py --dataset bbh_test
python scripts/generate_cot.py --dataset bb_sub_test
```

Queries `gpt-4.1` for several completions per question and keeps the first one containing the `"Therefore, the answer is"` marker. This step costs API credits. Pre-generated output is already committed under `data/generated_cots/`, so it is only needed to rebuild the dataset from scratch.

### 2. Train

```bash
python training/lora_train.py
```

Defaults: Llama-3.2-1B, LoRA `r=16` / `alpha=16` across all linear projections, 3 epochs, per-device batch size 2 with gradient accumulation 32, maximum sequence length 768. Adapters are written to `adapters/epoch-N/` at the end of every epoch.

### 3. Evaluate

```bash
python evaluation/evaluation.py
```

Set `saved_model_dir` to the adapter checkpoint and `test_dataset` to the benchmark to score. Available benchmarks are `bbh_eval_dataset` (in-domain) and `bb_eval_dataset`, `agieval_eval_dataset`, `arcc_eval_dataset`, `arce_eval_dataset` (out-of-domain). vLLM is used when installed and falls back to `transformers.generate()` otherwise — identical outputs, slower.

### 4. Interactive inference

```bash
python inference/run_lora_inference.py
```

A standalone CLI for querying the trained adapter by hand, with streaming generation, regex-based stopping criteria, and output modes `free`, `final`, `yesno`, `number`, `letter`, and `cot_yesno`. Intended for sanity-checking model behaviour rather than for scoring.

---

## Results

Benchmark results for this configuration are not yet published in this repo.

For context, the original paper reports Llama3-8B + CasCoD at 61.9 average accuracy against 52.7 for Std-CoT. Those figures are not directly comparable to this setup: the student here is roughly eight times smaller, the teacher differs, the training set is smaller and covers fewer tasks, and the cascading loss is not implemented.

---

## Roadmap

- [ ] Implement the cascading decomposed loss (two forward passes, `alpha`-weighted rationale and answer objectives) so the repo trains CasCoD rather than the Std-CoT baseline
- [ ] Ablate `alpha` at the 1B scale
- [ ] Extend teacher generation to the ten missing BBH tasks
- [ ] Publish IND and OOD evaluation numbers
- [ ] Pin a reproducible environment file

---

## Attribution and license

This repository reimplements and restructures [C-W-D/CasCoD](https://github.com/C-W-D/CasCoD), which is itself built on [llama-recipes](https://github.com/Meta-Llama/llama-recipes) and shared with the authors' related work [EDIT](https://github.com/C-W-D/EDIT). Released under the MIT license, matching upstream. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{dai-etal-2024-improve,
    title = "Improve Student's Reasoning Generalizability through Cascading Decomposed {C}o{T}s Distillation",
    author = "Dai, Chengwei and Li, Kun and Zhou, Wei and Hu, Songlin",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    pages = "15623--15643",
    doi = "10.18653/v1/2024.emnlp-main.875",
}
```
