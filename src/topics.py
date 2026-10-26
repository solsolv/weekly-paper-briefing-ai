"""관심 주제 분류 — 논문마다 config `interests.<track>` 목록 중 1~2개를 LLM이 고른다.

태그(자유 작성, 280여 종)와 달리 정해진 주제 목록에서만 고르므로, 사이트의 관심 주제
칩을 그대로 필터로 쓸 수 있다. 트랙별로 묶어 한 번의 호출로 여러 편을 분류해 호출 수를 최소화.
"""
from __future__ import annotations

import re
import time

from .config import Config
from .providers import LLMProvider

SYSTEM_PROMPT = (
    "당신은 논문을 미리 정해진 주제 목록으로 분류하는 큐레이터입니다. "
    "제목과 요약만 보고 가장 잘 맞는 주제 번호를 고르며, 목록 밖의 답이나 설명은 쓰지 않습니다."
)

PROMPT_TEMPLATE = """다음 논문들을 아래 주제 목록으로 분류하세요.

[주제 목록]
{topic_lines}

[논문]
{paper_lines}

규칙:
- 주제 이름보다 '—' 뒤의 설명 범위를 기준으로 판단합니다. 예) 재료 모델·유한요소 가속은 진단이 아니라 시뮬레이션입니다.
- 논문마다 가장 잘 맞는 주제 번호를 1개 고르고, 두 주제에 뚜렷하게 걸칠 때만 2개까지 고릅니다(주된 주제를 먼저).
- 어느 주제에도 잘 맞지 않아도 가장 가까운 1개를 고릅니다.
- 아래 형식으로 논문 수만큼 한 줄씩만 출력합니다. 다른 말은 쓰지 않습니다.

P<논문번호>: <주제번호>[,<주제번호>]
예) P1: 2
예) P2: 1,3
"""

_LINE_RE = re.compile(r"P\s*(\d+)\s*[:：]\s*([\d\s,]+)", re.I)
_BATCH = 25          # 한 호출당 최대 편수 (출력 토큰·정확도 균형)
_MAX_TOPICS = 2
_CALL_GAP_SEC = 13   # 호출 간 간격 — Gemini 무료 등급 분당 5회 제한 회피
_RATE_WAIT_SEC = 45


def _paper_text(item: dict) -> str:
    title = item.get("title_ko") or item.get("title", "")
    body = (item.get("summary_ko") or item.get("summary") or "").replace("\n", " ")
    return f"{title} / {item.get('title', '')} — {body[:400]}"


def _classify_batch(
    provider: LLMProvider, topics: list[str], hints: dict[str, str], batch: list[dict]
) -> dict[int, list[str]]:
    prompt = PROMPT_TEMPLATE.format(
        topic_lines="\n".join(
            f"{i}. {t}" + (f" — {hints[t]}" if hints.get(t) else "") for i, t in enumerate(topics, 1)
        ),
        paper_lines="\n".join(f"P{i}. {_paper_text(it)}" for i, it in enumerate(batch, 1)),
    )
    prev_max = getattr(provider, "max_tokens", None)
    raw = ""
    try:
        if prev_max is not None:
            provider.max_tokens = max(prev_max, 1500)
        for attempt in range(2):
            try:
                raw = provider.generate(SYSTEM_PROMPT, prompt)
                break
            except Exception as exc:  # noqa: BLE001
                # 무료 등급 분당 호출 제한(429) → 한 번만 기다렸다 재시도
                if attempt == 0 and "429" in str(exc):
                    print(f"    [info] 호출 제한(429) — {_RATE_WAIT_SEC}초 대기 후 재시도")
                    time.sleep(_RATE_WAIT_SEC)
                    continue
                print(f"    [warn] 주제 분류 호출 실패: {str(exc)[:200]}")
                return {}
    finally:
        if prev_max is not None:
            provider.max_tokens = prev_max

    out: dict[int, list[str]] = {}
    for m in _LINE_RE.finditer(raw or ""):
        idx = int(m.group(1)) - 1
        picked: list[str] = []
        for tok in m.group(2).split(","):
            tok = tok.strip()
            if tok.isdigit() and 1 <= int(tok) <= len(topics):
                t = topics[int(tok) - 1]
                if t not in picked:
                    picked.append(t)
        if 0 <= idx < len(batch) and picked:
            out[idx] = picked[:_MAX_TOPICS]
    return out


def assign_topics(cfg: Config, provider: LLMProvider, items: list[dict], *, force: bool = False) -> int:
    """items(논문 dict)에 'topics' 필드를 채운다(제자리 수정). 채운 편수 반환.

    force=False면 이미 topics가 있는 논문은 건너뛴다. 분류 실패한 논문은 필드를 비워 두어
    다음 빌드에서 다시 시도된다.
    """
    done = 0
    calls = 0
    hints = cfg.get("interests_hints", {}) or {}
    # 국내 논문(kr)은 기계공학 주제 목록에서 가장 가까운 주제로 분류
    for track, topic_key in (("ai", "ai"), ("mech", "mech"), ("kr", "mech")):
        topics = cfg.get(f"interests.{topic_key}", []) or []
        if not topics:
            continue
        todo = [it for it in items if it.get("track") == track and (force or not it.get("topics"))]
        for start in range(0, len(todo), _BATCH):
            batch = todo[start:start + _BATCH]
            if calls:
                time.sleep(_CALL_GAP_SEC)
            calls += 1
            print(f"  관심 주제 분류: {track} {len(batch)}편")
            result = _classify_batch(provider, topics, hints, batch)
            for i, it in enumerate(batch):
                if i in result:
                    it["topics"] = result[i]
                    done += 1
            missing = len(batch) - len(result)
            if missing:
                print(f"    [warn] {missing}편 분류 결과 없음 — 다음 빌드에서 재시도")
    return done
