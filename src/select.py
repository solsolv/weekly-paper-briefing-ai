"""슬롯별 선정 — 임팩트 1편(영향력 인용수 가중) + 최신 1편(발표일)."""
from __future__ import annotations

import json

from .collect import Paper
from .config import DATA_DIR, Config


def load_featured_ids() -> set[str]:
    """data/*.json 아카이브에 이미 등장한 arXiv ID 집합."""
    ids: set[str] = set()
    if not DATA_DIR.exists():
        return ids
    for fp in DATA_DIR.glob("*.json"):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for p in payload.get("papers", []):
            if p.get("arxiv_id"):
                ids.add(p["arxiv_id"])
    return ids


def impact_score(paper: Paper) -> tuple:
    """임팩트 정렬 키: influential_cc * 3 + citation_cc, tiebreak: 발표일."""
    icc = paper.influential_citation_count or 0
    cc = paper.citation_count or 0
    return (icc * 3 + cc, icc, cc, paper.published.timestamp())


def latest_score(paper: Paper) -> tuple:
    """최신 정렬 키: 발표일(최신 우선), tiebreak: HF upvote(있으면)."""
    return (paper.published.timestamp(), paper.hf_upvotes or 0)


# 모듈 내부 사용 alias (구버전 호환)
_impact_score = impact_score
_latest_score = latest_score


def select_impact(
    cfg: Config,
    candidates: list[Paper],
    featured_ids: set[str],
    exclude_ids: set[str] | None = None,
) -> Paper | None:
    exclude = bool(cfg.get("selection.exclude_already_featured", True))
    skip = (exclude_ids or set())
    pool = [
        p for p in candidates
        if not (exclude and p.arxiv_id in featured_ids) and p.arxiv_id not in skip
    ]
    pool.sort(key=_impact_score, reverse=True)
    return pool[0] if pool else None


def select_latest(
    cfg: Config,
    candidates: list[Paper],
    featured_ids: set[str],
    exclude_ids: set[str] | None = None,
) -> Paper | None:
    exclude = bool(cfg.get("selection.exclude_already_featured", True))
    skip = (exclude_ids or set())
    pool = [
        p for p in candidates
        if not (exclude and p.arxiv_id in featured_ids) and p.arxiv_id not in skip
    ]
    pool.sort(key=_latest_score, reverse=True)
    return pool[0] if pool else None


def drop_excluded(
    candidates: list[Paper],
    featured_ids: set[str],
    exclude_ids: set[str] | None = None,
) -> list[Paper]:
    """과거 소개 논문·중복 제외 대상을 후보 풀에서 미리 제거.

    게이트(top_k 채점)·trim 이전에 적용해야 한다. 나중에 빼면 impact 풀처럼 매주 상위권이
    비슷한 경우 게이트 통과작이 전부 기소개 논문이 되어 슬롯이 비어 버린다.
    """
    skip = featured_ids | (exclude_ids or set())
    if not skip:
        return candidates
    kept = [p for p in candidates if p.arxiv_id not in skip]
    if len(kept) != len(candidates):
        print(f"  기소개·중복 제외: {len(candidates) - len(kept)}편 → 후보 {len(kept)}편")
    return kept


def trim_candidate_pool(cfg: Config, candidates: list[Paper]) -> list[Paper]:
    """인용수 조회 비용을 줄이기 위해, 최신순 상위 N개만 후보로 남긴다."""
    limit = int(cfg.get("selection.candidate_pool_per_track", 150))
    return candidates[:limit]
