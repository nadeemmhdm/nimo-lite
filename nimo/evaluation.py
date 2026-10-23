"""Small, transparent behavioral diagnostics for trained Nimo chat checkpoints.

These probes are not a benchmark of general intelligence. They are deterministic smoke
checks that complement validation loss/perplexity and make obvious regressions visible.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    messages: tuple[tuple[str, str], ...]
    expected_any: tuple[str, ...] = ()
    expect_malayalam: bool = False
    max_new_tokens: int = 64


PROBES = (
    Probe(
        "conversation",
        (("user", "Hello. Briefly introduce yourself."),),
        ("nimo", "assistant", "help"),
        max_new_tokens=48,
    ),
    Probe(
        "multiturn_context",
        (
            ("user", "My project is called Nimo Garden."),
            ("assistant", "Understood."),
            ("user", "What is my project called? Answer with the project name."),
        ),
        ("nimo garden",),
        max_new_tokens=32,
    ),
    Probe(
        "malayalam",
        (("user", "മലയാളത്തിൽ ഒരു ചെറിയ അഭിവാദ്യം എഴുതുക."),),
        expect_malayalam=True,
        max_new_tokens=48,
    ),
    Probe(
        "manglish",
        (("user", "Python loop entha? Short ayi explain cheyyu."),),
        ("loop", "repeat", "for", "while"),
        max_new_tokens=64,
    ),
    Probe(
        "coding",
        (("user", "Write a short Python loop that prints numbers 0 to 2."),),
        ("for", "range", "while"),
        max_new_tokens=64,
    ),
    Probe(
        "math",
        (("user", "Calculate 2 + 3 * 4 and give the result."),),
        ("14",),
        max_new_tokens=32,
    ),
    Probe(
        "uncertainty",
        (("user", "You do not have current web data. What should you do if I ask for today's stock price?"),),
        ("verify", "current", "source", "uncertain", "cannot", "can't", "don't know"),
        max_new_tokens=64,
    ),
    Probe(
        "online_safety",
        (("user", "An unknown website asks for my password. What is a safe response?"),),
        ("password", "scam", "verify", "don't", "do not", "avoid"),
        max_new_tokens=64,
    ),
)


def _has_malayalam(text: str) -> bool:
    return any("\u0d00" <= char <= "\u0d7f" for char in text)


def evaluate_chat_behavior(model, tokenizer, *, limit_probes: int | None = None) -> dict:
    selected = PROBES[:limit_probes] if limit_probes else PROBES
    results = []
    for probe in selected:
        messages = [{"role": role, "content": content} for role, content in probe.messages]
        prompt_ids = tokenizer.encode_chat(messages, add_assistant_prompt=True)
        try:
            answer = model.generate(
                prompt_ids,
                tokenizer,
                limit=probe.max_new_tokens,
                deterministic=True,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.05,
            ).strip()
            lowered = answer.casefold()
            keyword_ok = not probe.expected_any or any(term.casefold() in lowered for term in probe.expected_any)
            language_ok = not probe.expect_malayalam or _has_malayalam(answer)
            passed = bool(answer) and keyword_ok and language_ok
            results.append({
                "name": probe.name,
                "passed": passed,
                "answer": answer[:2000],
                "expected_any": list(probe.expected_any),
                "expect_malayalam": probe.expect_malayalam,
            })
        except Exception as exc:
            results.append({"name": probe.name, "passed": False, "error": str(exc)[:300]})
    passed = sum(bool(row.get("passed")) for row in results)
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / max(len(results), 1),
        "probes": results,
        "interpretation": "Diagnostic smoke tests only; they do not establish general intelligence or factual reliability.",
    }
