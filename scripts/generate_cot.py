import argparse
import json
import os
from typing import Dict, Any, List


def _get_client():
    """Return an OpenAI client that exposes chat completions."""
    try:
        from openai import OpenAI  # >= 1.x
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set.")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if getattr(client, "chat", None) is None:
            raise RuntimeError("OpenAI client missing chat API.")
        return client, "chat"
    except Exception:
        import openai  # 0.x
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set.")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return openai, "legacy"


_CLIENT, _MODE = _get_client()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_prompt_template(task_name: str) -> str:
    cand = os.path.join(os.path.dirname(__file__), "..", "dataset", "bbh", "cot-prompts", f"{task_name}.txt")
    cand = os.path.normpath(cand)
    if os.path.exists(cand):
        with open(cand, "r", encoding="utf-8") as f:
            return f.read()
    return (
        "Answer the following multiple-choice question. Let's think step by step. "
        "Conclude with: Therefore, the answer is (X).\n\n{QUESTION}\n"
    )


def make_prompt(task_name: str, instruction: str) -> str:
    tpl = read_prompt_template(task_name)
    return tpl.replace("{QUESTION}", instruction)


def call_parent(messages, model: str, temperature: float, max_tokens: int, n: int):
    """Call parent model using chat APIs that support `n`."""
    if _MODE == "chat":  # modern OpenAI >= 1.x
        out = _CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return [choice.message.content for choice in out.choices]
    if hasattr(_CLIENT, "ChatCompletion"):  # legacy 0.x
        out = _CLIENT.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return [c["message"]["content"] for c in out["choices"]]
    raise RuntimeError(
        "OpenAI SDK does not expose chat.completions or ChatCompletion. "
        "Install a supported version: `pip install --upgrade openai`."
    )


def pick_with_marker(texts: List[str]) -> str:
    for t in texts:
        if "Therefore, the answer is" in t:
            return t
    return texts[0] if texts else ""


def main():
    ap = argparse.ArgumentParser(description="Generate CoTs from parent LLM and save JSON.")
    ap.add_argument("--dataset", choices=["bbh_train", "bbh_test", "bb_sub_test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent_model", default=os.getenv("PARENT_MODEL", "gpt-4.1"))
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if args.dataset == "bbh_train":
        src = os.path.join(repo_root, "dataset", "bbh", "bbh_all_data", "all_task_train_right_answer.json")
    elif args.dataset == "bbh_test":
        src = os.path.join(repo_root, "dataset", "bbh", "bbh_all_data", "all_task_test.json")
    else:
        src = os.path.join(repo_root, "dataset", "bb", "merged_data", "bb_sub_task_random100test.json")

    data = load_json(src)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        task_name = row.get("task_name", "generic_task")
        instruction = row.get("instruction", "")
        gt_output = row.get("output", "")
        task_desc = row.get("task_description", "")

        prompt = make_prompt(task_name, instruction)
        messages = [
            {"role": "system", "content": "Provide step-by-step reasoning and finish with: Therefore, the answer is (X)."},
            {"role": "user", "content": prompt},
        ]
        outputs = call_parent(messages, args.parent_model, args.temperature, args.max_tokens, args.n)
        response = pick_with_marker(outputs)

        out_rows.append({
            "instruction": instruction,
            "input": "",
            "output": gt_output,
            "prompt": prompt,
            "response": response,
            "task_description": task_desc,
            "task_name": task_name,
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(data)}")

    save_json(args.out, out_rows)
    print(f"Saved {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
