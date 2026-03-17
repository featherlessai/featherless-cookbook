"""
Featherless AI Model Benchmark
Measures TTFT (Time to First Token) and total completion time across models.

Usage:
    python benchmark.py
    python benchmark.py --models "meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-72B-Instruct"
    python benchmark.py --prompt "Explain quantum computing in simple terms"
    python benchmark.py --rounds 5
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import requests

FEATHERLESS_API_BASE = "https://api.featherless.ai/v1"

DEFAULT_MODELS = [
    "LyraNovaHeart/Stellar-Odyssey-12b-v0.0",
    "Qwen/Qwen3-235B-A22B",
    "openerotica/writing-roleplay-20k-context-nemo-12b-v1.0",
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3-0324",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen3-32B",
    "zai-org/GLM-5",
    "zai-org/GLM-4.7-Flash",
    "zai-org/GLM-4.6",
    "moonshotai/Kimi-K2.5",
    "moonshotai/Kimi-K2-Instruct-0905",
    "inclusionAI/Ling-1T",
    "MiniMaxAI/MiniMax-M2.5",
    "Nanbeige/Nanbeige4.1-3B",
    "Qwen/Qwen3.5-397B-A17B",
    "stepfun-ai/Step-3.5-Flash"
    
]

DEFAULT_PROMPT = """
    "Prompt

You are an AI assistant that writes thoughtful, engaging, and natural-sounding social media comments in response to posts. Your task is to read the provided context about a social media post and generate a single comment that would plausibly appear underneath that post. The goal is not to summarize the post, but to respond to it in a way that feels authentic, conversational, and appropriate for platforms such as X (Twitter), LinkedIn, Reddit, or Facebook.

The comment should sound like it was written by a real person. It should avoid sounding overly robotic, overly formal, or like an advertisement. The tone should be friendly, slightly informal, and aligned with the tone of the original post. If the post is professional or business-related, the comment should sound thoughtful and professional while still being conversational. If the post is casual or humorous, the comment may include light humor or enthusiasm.

You should follow these general writing principles when generating the comment:

The comment should be between 1 and 3 sentences.

It should be concise and easy to read.

It should directly reference something in the post so it feels contextual.

It should add a small amount of value (a thought, question, reaction, or insight).

It should avoid repeating the post verbatim.

It should not include hashtags unless the context clearly suggests them.

It should not include emojis unless the tone of the post is clearly casual or celebratory.

It should avoid generic responses like “Great post!” or “Thanks for sharing!” unless they are expanded with something more meaningful.

Think of the comment as something written by a knowledgeable and friendly member of an online community. The comment should show that the writer read the post and is reacting authentically.

When interpreting the context, consider the following elements:

The topic of the post

The tone of the post

Whether the post is sharing news, asking a question, celebrating an achievement, or expressing an opinion

Whether the appropriate response is agreement, curiosity, encouragement, or discussion

If the post is sharing a project, product, or technical idea, the comment might highlight an interesting aspect, ask a thoughtful question, or briefly share appreciation for the work. If the post is about a milestone or achievement, the comment should congratulate the person and potentially acknowledge the effort behind the accomplishment. If the post is opinionated or reflective, the comment may respond with agreement, a short reflection, or a constructive perspective.

Avoid writing anything that sounds promotional, spammy, or overly generic. The comment should feel organic and natural, as if written by someone scrolling through their feed and reacting in real time.

You must only output the comment itself. Do not include explanations, formatting, bullet points, or labels such as “Comment:” or “Response:”. The output should contain only the final comment text.

Here are some examples of the style of comments you should aim to produce. These examples illustrate tone and structure but should not be copied directly.

Example style 1:
“Really interesting approach to handling model inference at scale. Curious how it performs under heavy concurrent workloads?”

Example style 2:
“This is a great breakdown of the problem space. I especially like the way you explained the trade-offs between the different approaches.”

Example style 3:
“Congrats on the launch! Always exciting to see tools that make this workflow easier for developers.”

Example style 4:
“That’s a clever way to frame the issue. It’s surprising how often that pattern shows up in real-world systems.”

These examples demonstrate the desired balance between brevity, relevance, and authenticity.

Now read the following context representing a social media post and generate a single appropriate comment.

Context of the social media post:

A developer shares a post about experimenting with different open-source large language models hosted through a serverless inference platform. They explain that they were able to quickly switch between several Hugging Face models without managing infrastructure, which made it much easier to test prompts and compare outputs. The developer highlights how useful this approach is for rapid prototyping and mentions that developers can integrate these models into applications using a simple API. They also note that the ecosystem of open models is evolving quickly and that having flexible infrastructure makes experimentation much faster. The tone of the post is enthusiastic and focused on developer productivity and experimentation.

Generate one short social media comment responding to this post.
"""


QUALITY_RUBRIC_PROMPT = (
    "You are an impartial evaluator. Rate the following response on a scale of 1-10 "
    "for each criterion. Reply ONLY with valid JSON, no extra text.\n\n"
    "Criteria:\n"
    "- relevance: How well does it address the prompt?\n"
    "- accuracy: Is the information factually correct?\n"
    "- coherence: Is the writing clear and logically structured?\n"
    "- depth: Does it provide sufficient detail?\n\n"
    "Prompt given to the model:\n\"{prompt}\"\n\n"
    "Model response:\n\"{response}\"\n\n"
    'Respond with JSON like: {{"relevance": 8, "accuracy": 9, "coherence": 7, "depth": 6}}'
)


@dataclass
class BenchmarkResult:
    model: str
    ttft_seconds: float = 0.0
    total_seconds: float = 0.0
    tokens_generated: int = 0
    thinking_tokens: int = 0
    tokens_per_second: float = 0.0
    output: str = ""
    quality_scores: dict = field(default_factory=dict)
    error: str | None = None


def stream_completion(api_key: str, model: str, prompt: str, max_tokens: int = 4096) -> BenchmarkResult:
    """Send a streaming chat completion request and measure timing."""
    result = BenchmarkResult(model=model)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    try:
        start = time.perf_counter()
        first_token_time = None
        chunks: list[str] = []
        thinking_count = 0
        in_think_tag = False

        with requests.post(
            f"{FEATHERLESS_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                return result

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break

                data = json.loads(data_str)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")

                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    if "<think>" in content:
                        in_think_tag = True
                    if "</think>" in content:
                        in_think_tag = False
                        continue
                    if in_think_tag:
                        thinking_count += 1
                    else:
                        chunks.append(content)
                elif reasoning:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    thinking_count += 1

        end = time.perf_counter()

        result.output = "".join(chunks)
        result.total_seconds = end - start
        result.ttft_seconds = (first_token_time - start) if first_token_time else result.total_seconds
        result.tokens_generated = len(chunks)
        result.thinking_tokens = thinking_count
        total_tokens = len(chunks) + thinking_count
        generation_time = end - (first_token_time or start)
        result.tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

    except requests.exceptions.Timeout:
        result.error = "Request timed out (120s)"
    except Exception as e:
        result.error = str(e)

    return result


def evaluate_quality(api_key: str, judge_model: str, prompt: str, response_text: str) -> dict:
    """Use a model as a judge to score the response quality."""
    eval_prompt = QUALITY_RUBRIC_PROMPT.format(prompt=prompt, response=response_text)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": judge_model,
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": eval_prompt},
        ],
    }

    try:
        resp = requests.post(
            f"{FEATHERLESS_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            return {"error": f"Judge returned HTTP {resp.status_code}"}

        text = resp.json()["choices"][0]["message"]["content"].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"error": "Could not parse judge response"}
    except Exception as e:
        return {"error": str(e)}


def format_table(results: list[BenchmarkResult]) -> str:
    """Build an ASCII table of benchmark results."""
    header = f"{'Model':<50} {'TTFT (s)':>9} {'Total (s)':>10} {'Tok/s':>8} {'Tokens':>7} {'Think':>7}"
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for r in results:
        if r.error:
            lines.append(f"{r.model:<50} {'ERROR':>9}  {r.error[:40]}")
        else:
            think = str(r.thinking_tokens) if r.thinking_tokens else "—"
            lines.append(
                f"{r.model:<50} {r.ttft_seconds:>9.3f} {r.total_seconds:>10.3f} "
                f"{r.tokens_per_second:>8.1f} {r.tokens_generated:>7} {think:>7}"
            )

    lines.append(separator)

    has_quality = any(r.quality_scores and "error" not in r.quality_scores for r in results)
    if has_quality:
        lines.append("")
        q_header = f"{'Model':<50} {'Relevance':>10} {'Accuracy':>10} {'Coherence':>10} {'Depth':>10} {'Avg':>8}"
        q_sep = "-" * len(q_header)
        lines.extend([q_sep, q_header, q_sep])
        for r in results:
            qs = r.quality_scores
            if qs and "error" not in qs:
                avg = sum(qs.get(k, 0) for k in ("relevance", "accuracy", "coherence", "depth")) / 4
                lines.append(
                    f"{r.model:<50} {qs.get('relevance', '-'):>10} {qs.get('accuracy', '-'):>10} "
                    f"{qs.get('coherence', '-'):>10} {qs.get('depth', '-'):>10} {avg:>8.1f}"
                )
            elif qs:
                lines.append(f"{r.model:<50} {'judge error':>10}")
        lines.append(q_sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Featherless AI models")
    parser.add_argument("--api-key", default=os.getenv("FEATHERLESS_API_KEY"),
                        help="Featherless API key (or set FEATHERLESS_API_KEY env var)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="List of model IDs to benchmark")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Prompt to send to each model")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per request")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds per model (results are averaged)")
    parser.add_argument("--judge-model", default=None,
                        help="Model to use as quality judge (e.g. Qwen/Qwen2.5-72B-Instruct). Omit to skip quality eval.")
    parser.add_argument("--show-output", action="store_true",
                        help="Print each model's full response")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Provide an API key via --api-key or FEATHERLESS_API_KEY env var.")
        sys.exit(1)

    print("=" * 60)
    print("  Featherless AI — Model Benchmark")
    print("=" * 60)
    print(f"  Models : {len(args.models)}")
    print(f"  Rounds : {args.rounds}")
    print(f"  Tokens : {args.max_tokens} max")
    print(f"  Judge  : {args.judge_model or 'disabled'}")
    print(f"  Prompt : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print("=" * 60)
    print()

    all_results: list[BenchmarkResult] = []

    for model in args.models:
        round_results: list[BenchmarkResult] = []

        for r in range(1, args.rounds + 1):
            tag = f"[{r}/{args.rounds}]" if args.rounds > 1 else ""
            print(f"  Benchmarking {model} {tag}...", end=" ", flush=True)
            result = stream_completion(args.api_key, model, args.prompt, args.max_tokens)

            if result.error:
                print(f"ERROR: {result.error}")
            else:
                think_str = f" | Think {result.thinking_tokens}" if result.thinking_tokens else ""
                print(f"TTFT {result.ttft_seconds:.3f}s | Total {result.total_seconds:.3f}s | "
                      f"{result.tokens_per_second:.1f} tok/s{think_str}")
                print(f"\n  Response:\n  {'-' * 56}")
                for line in result.output.splitlines():
                    print(f"  {line}")
                print(f"  {'-' * 56}\n")
            round_results.append(result)

        successful = [r for r in round_results if not r.error]
        if not successful:
            avg = round_results[0]
        elif len(successful) == 1:
            avg = successful[0]
        else:
            avg = BenchmarkResult(
                model=model,
                ttft_seconds=sum(r.ttft_seconds for r in successful) / len(successful),
                total_seconds=sum(r.total_seconds for r in successful) / len(successful),
                tokens_generated=round(sum(r.tokens_generated for r in successful) / len(successful)),
                thinking_tokens=round(sum(r.thinking_tokens for r in successful) / len(successful)),
                tokens_per_second=sum(r.tokens_per_second for r in successful) / len(successful),
                output=successful[-1].output,
            )

        if args.judge_model and avg.output and not avg.error:
            print(f"  Evaluating quality with {args.judge_model}...", end=" ", flush=True)
            avg.quality_scores = evaluate_quality(args.api_key, args.judge_model, args.prompt, avg.output)
            if "error" in avg.quality_scores:
                print(f"WARN: {avg.quality_scores['error']}")
            else:
                score_avg = sum(avg.quality_scores.get(k, 0) for k in ("relevance", "accuracy", "coherence", "depth")) / 4
                print(f"avg score {score_avg:.1f}/10")

        all_results.append(avg)
        print()

    print("\n  Results" + (" (averaged over {} rounds)".format(args.rounds) if args.rounds > 1 else ""))
    print()
    print(format_table(all_results))

    if args.show_output:
        print("\n\n  Full Model Outputs")
        print("=" * 60)
        for r in all_results:
            print(f"\n--- {r.model} ---")
            print(r.output if r.output else "(no output)")
            print()


if __name__ == "__main__":
    main()
