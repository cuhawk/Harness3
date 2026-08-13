"""Cross-model answer verification: run one prompt across several models in parallel and
score how much their answers lexically agree, to catch a single model hallucinating or
overstating confidence rather than trusting its first answer.

AgentRuntime/list_proxy_models are imported lazily inside compare_models(), not at module
top level: tools_registry (which agent_runtime imports) assembles its tool list from every
tool module including this one, so a top-level `from .agent_runtime import ...` here would
close an import cycle at package load time.

The per-model runner is implemented locally rather than reused from subagent.py - that
module is being built concurrently by another task, so this file must not depend on its
internals (and does not need spawn_subagent's depth-guard/history-trim machinery anyway,
since each run here is a single independent one-shot invocation, not a delegated sub-task).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.tools import tool

from . import completeness
from . import core
from . import rag


def _run_one_model(model_name: str, prompt: str) -> dict[str, Any]:
    """Fresh AgentRuntime per model, one-shot invoke, last non-empty AIMessage as the answer -
    same answer-extraction walk spawn_subagent.py uses, kept local per this module's docstring."""
    try:
        from .agent_runtime import AgentRuntime

        runtime = AgentRuntime(model_name)
        result = runtime.invoke(prompt)
        final_text = ""
        for msg in reversed(result.get("messages", [])):
            content = getattr(msg, "content", None)
            if content and msg.__class__.__name__ == "AIMessage":
                final_text = str(content)
                break
        return {"model": model_name, "ok": True, "answer": core.trim_text(final_text, 4000), "error": None}
    except Exception as exc:
        return {"model": model_name, "ok": False, "answer": "", "error": repr(exc)}


@tool
def verify_finding(
    claim: str,
    evidence: str = "",
    target: str = "",
    red_model: str = "",
    blue_model: str = "",
    judge_model: str = "",
) -> str:
    """Adversarially triage a suspected security finding before you act on or report it.

    compare_models only measures whether models AGREE - and models agree on wrong things, so a
    high agreement score says nothing about whether a vulnerability is real. This runs a
    different shape: a red sub-agent is told to DISPROVE the claim (find the sanitizer, the
    auth check, the framework escaping, the reason the sink is unreachable), a blue sub-agent
    argues it holds, and a judge rules on the two arguments. The adversarial framing is the
    whole point - a model asked "is this real?" tends to agree with you, while a model asked
    "prove this is a false positive" goes looking for the control that kills it.

    Both sides have full tool access to this workspace, so they argue from the actual code
    rather than from plausibility. claim should be one specific assertion ("the search param on
    /api/v2/report is reflected unescaped into the HTML response"), and evidence should carry
    the file:line pointers or the request/response you are basing it on.

    Returns a verdict (real / false_positive / unproven), what the red team actually found, and
    the residual risk, plus status/coverage for the review itself - a partial status means one of
    the reviewers argued from capped or truncated evidence, so weigh the verdict accordingly. Use
    it before writing a report, not to find bugs."""
    try:
        from . import subagent  # lazy: subagent imports are heavy, see module docstring

        claim = (claim or "").strip()
        if not claim:
            return core.to_json({"ok": False, "error": "claim is required"})

        context = f"TARGET: {target}\n" if target.strip() else ""
        context += f"CLAIM UNDER TEST:\n{core.trim_text(claim, 3000)}\n"
        if evidence.strip():
            context += f"\nEVIDENCE OFFERED:\n{core.trim_text(evidence, 6000)}\n"

        red_task = (
            "You are a skeptical application-security reviewer whose ONLY job is to DISPROVE the "
            "claim below. Assume it is a false positive until the code forces you to conclude "
            "otherwise. Use your tools to read the actual code in this workspace.\n\n"
            f"{context}\n"
            "Hunt specifically for: input validation or sanitization on the path; output encoding "
            "or framework auto-escaping; an authentication, authorization, or CSRF check; the sink "
            "being unreachable, dead, or not attacker-controlled; a WAF/CSP/header mitigation; the "
            "value being trusted-internal rather than user-supplied.\n\n"
            "Reply with ONLY this JSON: {\"refuted\": true|false, \"controls_found\": [\"...\"], "
            "\"reasoning\": \"<= 6 sentences citing file:line where possible\"}"
        )
        blue_task = (
            "You are an exploit developer arguing that the claim below is a REAL, exploitable "
            "issue. Use your tools to read the actual code in this workspace and ground every "
            "point in it - do not speculate.\n\n"
            f"{context}\n"
            "Establish concretely: that the input is attacker-controlled, that it reaches the sink "
            "unmodified, that no effective control stops it, and what an attacker gains.\n\n"
            "Reply with ONLY this JSON: {\"holds\": true|false, \"attack_path\": \"<concrete steps>\", "
            "\"impact\": \"<what an attacker gains>\", \"reasoning\": \"<= 6 sentences citing file:line>\"}"
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            red_future = pool.submit(
                subagent._run_subagent_task, red_task, red_model, 6000,
                entry_point="verify_finding_red",
            )
            blue_future = pool.submit(
                subagent._run_subagent_task, blue_task, blue_model, 6000,
                entry_point="verify_finding_blue",
            )
            red, blue = red_future.result(), blue_future.result()

        red_text = str(red.get("result") or red.get("error") or "")
        blue_text = str(blue.get("result") or blue.get("error") or "")

        judge_task = (
            "You are adjudicating whether a security finding is real. Two reviewers examined the "
            "same code with opposite mandates. Weigh the ARGUMENTS AND CODE CITATIONS, not the "
            "confidence of the writing. A specific cited control beats a general assertion that "
            "something is exploitable, and vice versa. If neither side is grounded in the actual "
            "code, the verdict is 'unproven'.\n\n"
            f"{context}\n"
            f"REVIEWER ARGUING IT IS A FALSE POSITIVE:\n{core.trim_text(red_text, 5000)}\n\n"
            f"REVIEWER ARGUING IT IS REAL:\n{core.trim_text(blue_text, 5000)}\n\n"
            "Reply with ONLY this JSON: {\"verdict\": \"real\"|\"false_positive\"|\"unproven\", "
            "\"confidence\": \"high\"|\"medium\"|\"low\", \"decisive_factor\": \"<one sentence>\", "
            "\"residual_risk\": \"<what would still need checking>\", "
            "\"recommended_next_step\": \"<one concrete action>\"}"
        )
        verdict = subagent._run_subagent_task(
            judge_task, judge_model, 4000,
            validator=subagent.build_validator(require_json=True, require_keys=["verdict"]),
            max_attempts=2,
            validator_kinds=["require_json", "require_keys"],
            entry_point="verify_finding_judge",
        )

        parsed: dict[str, Any] | None = None
        try:
            import json as _json

            parsed = _json.loads(subagent._strip_code_fence(str(verdict.get("result") or "")))
        except Exception:
            parsed = None

        # The three inner runs each computed a completeness verdict; dropping them here would make
        # this - the highest-stakes delegated output in the delegate preset, the one a report gets
        # written from - the only one with no coverage information at all. A 'false_positive'
        # reached because the red reviewer's grep returned 50 of 900 matches must not be
        # indistinguishable from one reached on complete evidence.
        review_signals: list[dict[str, Any]] = []
        for run in (red, blue, verdict):
            review_signals.extend((run.get("coverage") or {}).get("signals") or [])
        if not (red.get("ok") and blue.get("ok")):
            review_signals.append({
                "tool": "verify_finding", "reason": "reviewer_run_failed",
                "returned": sum(1 for r in (red, blue) if r.get("ok")), "total": 2,
            })

        return core.to_json({
            "ok": True,
            "claim": core.trim_text(claim, 500),
            "target": target,
            "verdict": (parsed or {}).get("verdict", "unproven"),
            **completeness.envelope(True, review_signals),
            "judgement": parsed if parsed is not None else core.trim_text(str(verdict.get("result")), 2000),
            "red_team": {"model": red.get("model"), "ok": red.get("ok"), "argument": core.trim_text(red_text, 2500)},
            "blue_team": {"model": blue.get("model"), "ok": blue.get("ok"), "argument": core.trim_text(blue_text, 2500)},
            "note": (
                "A 'real' verdict is a triage signal, not a proof of exploitability - reproduce it "
                "live before reporting."
            ),
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def compare_models(prompt: str, models_json: str = "", n: int = 3) -> str:
    """Run the same prompt independently across multiple models concurrently and cross-check
    their answers for lexical agreement, to reduce hallucinated/overconfident single-model
    answers. models_json, if non-empty, must be a JSON array of exact model name strings to
    use verbatim; otherwise up to n (clamped 1-6) models are auto-discovered from the LiteLLM
    proxy. Returns per-model answers plus pairwise agreement scores, with a low-agreement
    warning note when any pair scores roughly below one third."""
    try:
        from .agent_runtime import AgentRuntime, list_proxy_models  # lazy: see module docstring

        prompt_trimmed = core.trim_text(prompt, 4000)

        models_json = (models_json or "").strip()
        if models_json:
            requested = core.parse_json_arg(models_json, [])
            if not isinstance(requested, list):
                return core.to_json({"ok": False, "error": "models_json must be a JSON array of model name strings"})
            candidates = [str(m).strip() for m in requested if str(m).strip()]
            if not candidates:
                return core.to_json({"ok": False, "error": "models_json parsed to an empty list of model names"})
        else:
            count = max(1, min(int(n), 6))
            try:
                discovered = list_proxy_models()
            except Exception as exc:
                return core.to_json({"ok": False, "error": f"Could not discover proxy models: {repr(exc)}"})
            candidates = discovered[:count]
            if not candidates:
                return core.to_json({"ok": False, "error": "No models available from list_proxy_models()"})

        fell_back_to_few = len(set(candidates)) < 2

        answers: list[dict[str, Any] | None] = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            # Submit all first, then collect in the original candidate order (not
            # completion order), per the ordering requirement in the design.
            futures = [pool.submit(_run_one_model, name, prompt_trimmed) for name in candidates]
            for i, fut in enumerate(futures):
                answers[i] = fut.result()

        pairwise: list[dict[str, Any]] = []
        min_score: float | None = None
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                a, b = answers[i], answers[j]
                # rag_lexical_score(query, text_value) is asymmetric (overlap / len(query
                # tokens)), so average both directions for a direction-independent
                # agreement score between the two answers.
                fwd = rag.rag_lexical_score(a["answer"], b["answer"])
                bwd = rag.rag_lexical_score(b["answer"], a["answer"])
                score = round((fwd + bwd) / 2.0, 4)
                pairwise.append({"model_a": a["model"], "model_b": b["model"], "agreement_score": score})
                min_score = score if min_score is None else min(min_score, score)

        out: dict[str, Any] = {
            "ok": True,
            "prompt": prompt_trimmed,
            "models": candidates,
            "fell_back_to_fewer_than_two_distinct_models": fell_back_to_few,
            "answers": answers,
            "pairwise_agreement": pairwise,
        }
        if min_score is not None and min_score < (1.0 / 3.0):
            out["note"] = (
                "Low agreement between at least one model pair suggests at least one model is "
                "unreliable here; re-examine the underlying tool evidence before trusting either answer."
            )
        return core.to_json(out)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
