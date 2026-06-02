"""LLM 적합도 게이트 — 후보 풀을 사용자 선호 주제 기준으로 점수화·필터링.

키워드 필터를 통과한 정렬된 후보(상위 K편)를 LLM에 한 번 보내 0~1 점수를 받고,
임계값 이상만 다음 단계(`select_impact` / `select_latest`)로 넘긴다.

게이트 응답이 파싱되지 않거나 모두 임계값 미만이면 `empty_pool_behavior`에 따라
fallback 처리한다.
"""
from __future__ import annotations

import json
import re

from .collect import Paper
from .config import Config
from .providers import LLMProvider

SYSTEM_PROMPT = (
    "당신은 한국어 사용자의 학술 논문 큐레이션을 보조하는 분류기입니다. "
    "지정된 선호 주제 기준에 따라 각 논문을 정확히 0~1 점수로 평가하고, "
    "지시된 JSON 스키마로만 응답합니다."
)

_JSON_RE = re.compile(r"\{[\s\S]*\}")
# 표준 파싱 실패 시 fallback: "idx": N, "score": X 쌍을 직접 복구 (잘림·따옴표 깨짐 내성)
_ENTRY_RE = re.compile(r'"idx"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*([-+]?\d*\.?\d+)')
# 말미 콤마(trailing comma) 보정용
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _extract_scores(raw: str) -> tuple[list[dict], str]:
    """LLM 응답에서 점수 항목을 최대한 복구.

    반환: (entries, mode). mode ∈ {"strict","repaired","regex","none"}
    - strict : 표준 JSON 파싱 성공
    - repaired: 말미 콤마 제거 후 파싱 성공
    - regex  : JSON 파싱 실패 → idx/score 쌍만 정규식으로 복구 (출력 잘림 등에 내성)
    - none   : 아무것도 복구 못 함
    """
    match = _JSON_RE.search(raw)
    blob = match.group(0) if match else raw

    for candidate in (blob, _TRAILING_COMMA_RE.sub(r"\1", blob)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        scores = data.get("scores") if isinstance(data, dict) else None
        if isinstance(scores, list):
            return scores, ("strict" if candidate == blob else "repaired")

    # fallback은 brace-trim된 blob이 아니라 원본 전체를 스캔한다.
    # 출력이 잘리면 마지막 완성된 '}'가 앞쪽에 있어, 그 뒤의 정상 idx/score 쌍이
    # blob에서 누락되기 때문. raw 전체를 보면 잘린 꼬리의 점수까지 복구된다.
    entries = [
        {"idx": int(m.group(1)), "score": float(m.group(2))}
        for m in _ENTRY_RE.finditer(raw)
    ]
    return entries, ("regex" if entries else "none")


def _abstract_snippet(p: Paper, max_chars: int = 380) -> str:
    text = (p.summary or "").strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def _build_prompt(description: str, candidates: list[Paper]) -> str:
    blocks = []
    for i, p in enumerate(candidates):
        blocks.append(f"[{i}] {p.title}\n    {_abstract_snippet(p)}")
    listing = "\n\n".join(blocks)
    return f"""사용자의 선호 주제는 다음과 같습니다.

{description.strip()}

아래 후보 논문 {len(candidates)}편 각각에 대해 위 기준을 엄격히 적용해 0~1 점수로 평가하세요.

[평가 척도]
- 0.8~1.0: 핵심 영역에 명확히 부합하고, abstract에서 **구체적 물리 응용 도메인**이 명시됨
- 0.5~0.7: 인접 영역, 응용 도메인은 명시되나 사용자 핵심에서 비껴남
- 0.2~0.4: 응용 도메인이 모호하거나 ML 일반 알고리즘 응용
- 0.0~0.1: 명시된 제외 분야, 또는 ML/수학 이론 자체 연구

[필수 검증]
- abstract에 구체적 물리 응용 도메인(어떤 유체·구조·기계 시스템)이 명시되었는지 먼저 확인.
- 명시되지 않았다면 표면적 키워드(surrogate, neural operator 등)와 무관하게 0.2 이하.
- reason 필드는 **12자 이내**로 abstract의 응용 도메인을 짧게 적거나(예: "베어링 진단"),
  명시되지 않았다면 "도메인 미명시"로 표기. 따옴표(")는 쓰지 마세요.

다음 JSON 스키마로만 응답하세요. 머리말·맺음말·코드펜스·주석 금지.
{{"scores": [{{"idx": 0, "score": 0.85, "reason": "12자 이내"}}, ...]}}

[후보 논문]
{listing}
"""


def score_candidates(
    provider: LLMProvider,
    description: str,
    candidates: list[Paper],
    max_tokens: int | None = None,
) -> dict[int, tuple[float, str]]:
    """후보별 (idx → (score, reason)) 매핑. 호출/파싱 실패 시 빈 dict.

    max_tokens가 주어지면 게이트 호출에 한해 provider의 출력 상한을 일시 상향한다
    (요약용 상한은 건드리지 않음). 후보 수만큼의 점수 항목이 잘리지 않도록 보장.
    """
    if not candidates:
        return {}
    prompt = _build_prompt(description, candidates)
    prev_max = getattr(provider, "max_tokens", None)
    try:
        if max_tokens is not None and prev_max is not None:
            provider.max_tokens = max_tokens
        raw = provider.generate(SYSTEM_PROMPT, prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] relevance gate 호출 실패: {exc}")
        return {}
    finally:
        if max_tokens is not None and prev_max is not None:
            provider.max_tokens = prev_max
    if not raw:
        return {}

    entries, mode = _extract_scores(raw)
    if mode == "repaired":
        print("    [info] relevance gate JSON 말미 콤마 보정 후 파싱")
    elif mode == "regex":
        print(f"    [warn] relevance gate JSON 파싱 실패 — 정규식으로 {len(entries)}개 항목 복구")
    elif mode == "none":
        print(f"    [warn] relevance gate 점수 추출 실패 — head 120자: {raw[:120]!r}")
        return {}

    out: dict[int, tuple[float, str]] = {}
    for entry in entries:
        try:
            idx = int(entry["idx"])
            score = float(entry["score"])
        except (KeyError, ValueError, TypeError):
            continue
        reason = str(entry.get("reason", "")).strip()
        out[idx] = (max(0.0, min(1.0, score)), reason)
    return out


def filter_by_relevance(
    cfg: Config,
    provider: LLMProvider,
    track: str,
    candidates_sorted: list[Paper],
) -> list[Paper]:
    """이미 정렬된 후보 풀의 상위 top_k 편을 LLM 게이트로 필터링.

    - 게이트 비활성 또는 트랙 미적용 → 원본 그대로 반환.
    - 게이트 호출·파싱 실패 → 안전하게 원본 그대로 반환.
    - 모두 임계값 미만 → `empty_pool_behavior` 설정에 따라:
        "fallback_top1": 원본 상위 1편만 반환
        "skip":          빈 리스트 반환
    """
    if not cfg.get("relevance_gate.enabled", False):
        return candidates_sorted
    apply = cfg.get("relevance_gate.apply_to_tracks", []) or []
    if apply and track not in apply:
        return candidates_sorted
    if not candidates_sorted:
        return candidates_sorted

    k = int(cfg.get("relevance_gate.top_k", 10))
    threshold = float(cfg.get("relevance_gate.threshold", 0.5))
    gate_max_tokens = int(cfg.get("relevance_gate.max_output_tokens", 3000))
    description = cfg.get("relevance_gate.description", "") or ""
    behavior = cfg.get("relevance_gate.empty_pool_behavior", "fallback_top1")
    required_cats = cfg.get("relevance_gate.required_categories", []) or []

    if not description.strip():
        return candidates_sorted

    if required_cats:
        cats_set = set(required_cats)
        filtered = [
            p for p in candidates_sorted
            if any(c in cats_set for c in (p.categories or []))
        ]
        if filtered:
            print(
                f"  카테고리 사전 필터: {len(filtered)}/{len(candidates_sorted)}편 통과 "
                f"(요건: {', '.join(required_cats)})"
            )
            candidates_sorted = filtered
        else:
            print(
                f"    [warn] 카테고리 사전 필터 통과 0편 — 필터 우회, 원본 풀 유지 "
                f"(요건: {', '.join(required_cats)})"
            )

    head = candidates_sorted[:k]
    print(f"  LLM 적합도 게이트 평가 ({len(head)}편, threshold={threshold})...")
    scores = score_candidates(provider, description, head, max_tokens=gate_max_tokens)
    if not scores:
        print("    [warn] 점수 없음 — 게이트 우회, 원본 풀 유지")
        return candidates_sorted

    passed: list[Paper] = []
    for i, p in enumerate(head):
        s, r = scores.get(i, (0.0, "(미평가)"))
        if s >= threshold:
            p.matched_keywords = (p.matched_keywords or []) + [f"relevance:{s:.2f}"]
            passed.append(p)
            mark = "✓"
        else:
            mark = "✗"
        print(f"    [{mark} {s:.2f}] {p.title[:62]}  — {r[:60]}")

    if passed:
        return passed
    if behavior == "fallback_top1":
        print("    [warn] 모두 임계값 미만 — 원본 1위로 fallback")
        return candidates_sorted[:1]
    print("    [warn] 모두 임계값 미만 — 슬롯 비움")
    return []
