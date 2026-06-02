"""엔트리포인트: 논문 수집(슬롯별) → 인용수·기관 → 선정 → 요약 → 사이트 생성.

사용 예:
    python -m src.main                  # 정상 실행 (4슬롯 통째 빌드 + 요약 + 렌더)
    python -m src.main --dry-run        # LLM 호출 없이 파이프라인만 점검
    python -m src.main --no-citations   # Semantic Scholar 호출 생략(최신순으로 선정)

슬롯 분리 실행(GitHub Actions에서 슬롯별 step 분리 / 실패 슬롯만 재진행용):
    python -m src.main --slot mech-impact   # 한 슬롯만 선정 후 staging 파일로 저장
    python -m src.main --assemble           # staging + 기존 결과 조립 → 요약 → 저장 → 렌더
    python -m src.main --print-week-id      # 현재 주차 ID만 출력(step 간 공유용)

슬롯 ID: ai-impact | ai-latest | mech-impact | mech-latest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_env_local() -> None:
    """프로젝트 루트의 `.env.local`을 KEY=VALUE 형식으로 환경변수에 주입.
    이미 셸에 설정된 키는 덮어쓰지 않는다 (CI/Actions의 secrets 우선)."""
    fp = Path(__file__).resolve().parent.parent / ".env.local"
    if not fp.exists():
        return
    for line in fp.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env_local()

from .collect import (
    Paper,
    collect_candidates,
    enrich_citations,
    fetch_arxiv_by_ids,
    fetch_huggingface_weekly,
)
from .config import DATA_DIR, Config
from .relevance import filter_by_relevance
from .render import TRACK_LABEL, render_site, save_week_data
from .select import (
    impact_score,
    latest_score,
    load_featured_ids,
    select_impact,
    select_latest,
    trim_candidate_pool,
)

# 슬롯 분리 실행에서 사용하는 표준 슬롯 ID와 처리 순서(렌더 정렬에도 사용).
SLOT_ORDER: list[tuple[str, str]] = [
    ("ai", "impact"),
    ("ai", "latest"),
    ("mech", "impact"),
    ("mech", "latest"),
]
SLOT_IDS = [f"{t}-{k}" for t, k in SLOT_ORDER]
_STAGING_DIRNAME = "_staging"


def _tz(name: str):
    try:
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - tzdata 미설치 등
        if name == "Asia/Seoul":
            return timezone(timedelta(hours=9))
        return timezone.utc


def compute_week_id(tz_name: str) -> str:
    """공개 대상인 '다가오는(또는 오늘인) 월요일' 날짜를 YYYY-MM-DD로 반환 (배포 타임존 기준).

    토/일에 미리 빌드해 다음 월요일 08:00에 공개하는 운영을 위해, 지난 월요일이 아니라
    아직 오지 않은 월요일을 주차 ID로 삼는다.
    예) 토(05-30)→06-01, 일(05-31)→06-01, 월(06-01)→06-01.
    토요일 스케줄 기준 GitHub 지연(<~1일)이 있어도 일요일 안에 실행되어 같은 월요일로 수렴.
    과거 주차 재빌드는 --week-id로 명시 지정.
    """
    now_local = datetime.now(_tz(tz_name))
    days_ahead = (7 - now_local.weekday()) % 7  # 다음 월요일까지 남은 일수(오늘이 월요일이면 0)
    monday = now_local + timedelta(days=days_ahead)
    return monday.strftime("%Y-%m-%d")


def _parse_slot_id(slot_id: str) -> tuple[str, str]:
    """'mech-impact' → ('mech', 'impact'). 유효하지 않으면 ('', '')."""
    track, _, kind = slot_id.partition("-")
    if f"{track}-{kind}" not in SLOT_IDS:
        return "", ""
    return track, kind


# --- 슬롯 staging 입출력 ------------------------------------------------------
def _staging_dir(week_id: str) -> Path:
    return DATA_DIR / _STAGING_DIRNAME / week_id


def _staging_path(week_id: str, slot_id: str) -> Path:
    return _staging_dir(week_id) / f"{slot_id}.json"


def _write_staging(week_id: str, slot_id: str, track: str, paper: Paper | None) -> Path:
    d = _staging_dir(week_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "slot_id": slot_id,
        "track": track,
        "status": "selected" if paper is not None else "empty",
        "paper": paper.to_dict() if paper is not None else None,
    }
    fp = _staging_path(week_id, slot_id)
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def _read_staging(week_id: str, slot_id: str) -> dict | None:
    """해당 슬롯 staging을 읽어 반환. 파일 없으면 None (이번 실행에서 안 돌린 슬롯)."""
    fp = _staging_path(week_id, slot_id)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] {fp.name} staging 로드 실패: {exc}")
        return None


def _placeholder_summary(paper: Paper, n_lines: int) -> tuple[str, str, list[str]]:
    bullets = ["- (dry-run 모드: 실제 요약은 생성되지 않았습니다.)"]
    bullets += [f"- 초록 발췌: {paper.summary[:60]}..."]
    bullets += ["- 자세한 내용은 원문 링크를 참고하세요."] * max(0, n_lines - 2)
    return paper.title, "\n".join(bullets[:n_lines]), ["dry-run"]


def _collect_latest_for_ai_via_hf(cfg: Config) -> list[Paper]:
    """AI 최신 슬롯 후보를 HF Papers trending에서 수집 → arXiv 본문 보강."""
    top_n = int(cfg.get("huggingface.top_n", 10))
    hf_items = fetch_huggingface_weekly(top_n=top_n, days_back=7)
    if not hf_items:
        print("  [info] HF Papers 결과 없음")
        return []
    arxiv_ids = [it["arxiv_id"] for it in hf_items]
    print(f"  HF Papers 후보 {len(arxiv_ids)}편 → arXiv 본문 보강")
    papers = fetch_arxiv_by_ids(arxiv_ids)
    # HF upvote 정보 매핑
    upv_map = {it["arxiv_id"]: it["upvotes"] for it in hf_items}
    for p in papers:
        p.source = "huggingface"
        p.track = "ai"
        p.hf_upvotes = upv_map.get(p.arxiv_id)
    return papers


# --- 슬롯 단위 선정 (모놀리식 / 분리 실행 공용) -------------------------------
def _run_impact_slot(
    cfg: Config,
    provider,
    track: str,
    featured: set[str],
    *,
    no_citations: bool,
) -> Paper | None:
    impact_lookback = int(cfg.get("selection.slots.impact.lookback_days", 365))
    impact_s2_sort = cfg.get("s2_search.sort_impact", "citationCount:desc")
    print(f"\n[{TRACK_LABEL[track]}] === 임팩트 슬롯 ({impact_lookback}일 lookback) ===")
    pool = collect_candidates(cfg, track, lookback_days=impact_lookback, s2_sort=impact_s2_sort)
    pool = trim_candidate_pool(cfg, pool)
    print(f"  후보 {len(pool)}편")
    if not no_citations:
        print("  인용수·소속기관 조회(Semantic Scholar)...")
        enrich_citations(cfg, pool)
    else:
        for c in pool:
            c.citation_count = 0
    pool.sort(key=impact_score, reverse=True)
    if provider is not None:
        pool = filter_by_relevance(cfg, provider, track, pool)
    pick = select_impact(cfg, pool, featured)
    if pick:
        pick.slot = "impact"
        cc = pick.citation_count if pick.citation_count is not None else "?"
        icc = pick.influential_citation_count if pick.influential_citation_count is not None else "?"
        print(f"  임팩트 선정: [cc={cc}, infl={icc}] {pick.title[:70]}")
    else:
        print("  임팩트 선정작 없음")
    return pick


def _run_latest_slot(
    cfg: Config,
    provider,
    track: str,
    featured: set[str],
    *,
    no_citations: bool,
    exclude_ids: set[str],
) -> Paper | None:
    latest_lookback = int(cfg.get("selection.slots.latest.lookback_days", 14))
    latest_source_cfg = cfg.get("selection.latest_source", {}) or {}
    hf_enabled = bool(cfg.get("huggingface.enabled", True))
    hf_fallback = bool(cfg.get("huggingface.fallback_to_arxiv", True))
    latest_s2_sort = cfg.get("s2_search.sort_latest", "publicationDate:desc")
    latest_source = latest_source_cfg.get(track, "arxiv")
    print(f"\n[{TRACK_LABEL[track]}] === 최신 슬롯 ({latest_lookback}일 lookback, source={latest_source}) ===")
    pool: list[Paper] = []
    if latest_source == "huggingface_weekly" and hf_enabled:
        pool = _collect_latest_for_ai_via_hf(cfg)
        if not pool and hf_fallback:
            print("  [info] HF 결과 없음 → arXiv 14일 fallback")
            pool = collect_candidates(cfg, track, lookback_days=latest_lookback, s2_sort=latest_s2_sort)
    else:
        pool = collect_candidates(cfg, track, lookback_days=latest_lookback, s2_sort=latest_s2_sort)
    pool = trim_candidate_pool(cfg, pool)
    print(f"  후보 {len(pool)}편")
    if not no_citations:
        print("  인용수·소속기관 조회(Semantic Scholar)...")
        enrich_citations(cfg, pool)
    else:
        for c in pool:
            c.citation_count = 0
    pool.sort(key=latest_score, reverse=True)
    if provider is not None:
        pool = filter_by_relevance(cfg, provider, track, pool)
    pick = select_latest(cfg, pool, featured, exclude_ids=exclude_ids)
    if pick:
        pick.slot = "latest"
        upv = f", hf_upvotes={pick.hf_upvotes}" if pick.hf_upvotes is not None else ""
        print(f"  최신 선정: [pub={pick.published.date()}{upv}] {pick.title[:70]}")
    else:
        print("  최신 선정작 없음")
    return pick


def _build_provider_verbose(cfg: Config):
    from .providers import build_provider  # noqa: PLC0415

    provider = build_provider(cfg)
    print(f"  provider={provider.name}, model={provider.model}")
    return provider


def _publish_at(cfg: Config, week_id: str) -> str:
    """주차(월요일)의 공개 시각을 배포 타임존 기준 ISO8601로 반환 (기본 08:00).

    사이트의 클라이언트 JS가 이 시각 전에는 해당 주차를 숨기고 정시에 노출한다 —
    GitHub Actions 스케줄 지연과 무관하게 브라우저 시계로 정확히 공개.
    week_id 형식이 이상하면 빈 문자열(=게이트 비활성, 즉시 노출)."""
    tz_name = cfg.get("site.timezone", "Asia/Seoul")
    hour = int(cfg.get("site.publish_hour", 8))
    try:
        y, m, d = (int(x) for x in week_id.split("-"))
        return datetime(y, m, d, hour, 0, 0, tzinfo=_tz(tz_name)).isoformat()
    except Exception:  # noqa: BLE001
        return ""


def _make_payload(cfg: Config, week_id: str, items: list[dict]) -> dict:
    tracks = ["ai", "mech"]
    return {
        "week_id": week_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "publish_at": _publish_at(cfg, week_id),
        "site_title": cfg.get("site.title", ""),
        "provider": cfg.get("llm.provider", ""),
        "impact_lookback_days": int(cfg.get("selection.slots.impact.lookback_days", 365)),
        "latest_lookback_days": int(cfg.get("selection.slots.latest.lookback_days", 14)),
        "tracks": {t: [it for it in items if it.get("track") == t] for t in tracks},
        "papers": items,
    }


# --- 실행 모드 1: 모놀리식 (기존 동작 유지) ----------------------------------
def run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    tz_name = cfg.get("site.timezone", "Asia/Seoul")
    week_id = args.week_id or compute_week_id(tz_name)
    print(f"== 주간 브리핑 빌드: {week_id} (tz={tz_name}) ==")

    featured = load_featured_ids() if cfg.get("selection.exclude_already_featured", True) else set()
    if featured:
        print(f"  과거 소개 논문 {len(featured)}편 제외 대상")

    provider = None if args.dry_run else _build_provider_verbose(cfg)

    selected: list[Paper] = []
    for idx, track in enumerate(["ai", "mech"]):
        if idx > 0:
            time.sleep(15)  # arXiv rate limit 회피용 트랙 간 대기
        impact_pick = _run_impact_slot(cfg, provider, track, featured, no_citations=args.no_citations)
        time.sleep(15)
        exclude_ids = {impact_pick.arxiv_id} if impact_pick else set()
        latest_pick = _run_latest_slot(
            cfg, provider, track, featured, no_citations=args.no_citations, exclude_ids=exclude_ids
        )
        for p in (impact_pick, latest_pick):
            if p is not None:
                selected.append(p)

    if not selected:
        print("\n선정된 논문이 없습니다. (쿼리/기간 설정을 확인하세요)")
        return 1

    print(f"\n총 {len(selected)}편 요약 시작...")
    n_lines = int(cfg.get("llm.summary_lines", 5))
    if args.dry_run:
        items: list[dict] = []
        for p in selected:
            _, summary, tags = _placeholder_summary(p, n_lines)
            d = p.to_dict()
            d["title_ko"], d["summary_ko"], d["tags"] = "", summary, tags
            items.append(d)
    else:
        from .summarize import summarize_all  # noqa: PLC0415

        items = summarize_all(cfg, provider, selected)

    fp = save_week_data(week_id, _make_payload(cfg, week_id, items))
    print(f"  데이터 저장: {fp}")
    render_site(cfg)
    print("\n완료.")
    return 0


# --- 실행 모드 2: 단일 슬롯 (수집·선정만, staging 저장) -----------------------
def run_single_slot(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    tz_name = cfg.get("site.timezone", "Asia/Seoul")
    week_id = args.week_id or compute_week_id(tz_name)
    track, kind = _parse_slot_id(args.slot)
    if not track:
        print(f"[error] 알 수 없는 슬롯 ID: {args.slot!r} (가능: {', '.join(SLOT_IDS)})")
        return 2
    print(f"== 슬롯 빌드: {args.slot} (week={week_id}, tz={tz_name}) ==")

    featured = load_featured_ids() if cfg.get("selection.exclude_already_featured", True) else set()
    if featured:
        print(f"  과거 소개 논문 {len(featured)}편 제외 대상")

    provider = None if args.dry_run else _build_provider_verbose(cfg)

    if kind == "impact":
        pick = _run_impact_slot(cfg, provider, track, featured, no_citations=args.no_citations)
    else:  # latest — 같은 트랙 impact 선정작을 staging에서 읽어 중복 제외
        exclude_ids: set[str] = set()
        imp = _read_staging(week_id, f"{track}-impact")
        if imp and imp.get("paper"):
            exclude_ids = {imp["paper"]["arxiv_id"]}
            print(f"  중복 제외: {track}-impact 선정작 1편")
        elif imp is None:
            print(f"  [info] {track}-impact staging 없음 — 중복 제외 생략")
        pick = _run_latest_slot(
            cfg, provider, track, featured, no_citations=args.no_citations, exclude_ids=exclude_ids
        )

    fp = _write_staging(week_id, args.slot, track, pick)
    print(f"  staging 저장: {fp}")
    return 0


# --- 실행 모드 3: 조립 (staging + 기존 결과 → 요약 → 저장 → 렌더) ------------
def run_assemble(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    tz_name = cfg.get("site.timezone", "Asia/Seoul")
    week_id = args.week_id or compute_week_id(tz_name)
    print(f"== 조립·렌더: {week_id} (tz={tz_name}) ==")

    # 직전 커밋된 주차 JSON을 base로 — 이번에 안 돌린 슬롯은 기존 요약을 그대로 재사용.
    base_by_slot: dict[tuple[str, str], dict] = {}
    existing_fp = DATA_DIR / f"{week_id}.json"
    if existing_fp.exists():
        try:
            base = json.loads(existing_fp.read_text(encoding="utf-8"))
            for it in base.get("papers", []):
                base_by_slot[(it.get("track", ""), it.get("slot", ""))] = it
            print(f"  기존 {existing_fp.name} 로드: {len(base_by_slot)}슬롯")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 기존 {existing_fp.name} 로드 실패: {exc}")

    final_by_slot: dict[tuple[str, str], dict] = {}
    to_summarize: list[Paper] = []

    for track, kind in SLOT_ORDER:
        slot_id = f"{track}-{kind}"
        st = _read_staging(week_id, slot_id)
        if st is not None:
            # 이번 실행에서 새로 돌린 슬롯 → 새 선정작으로 교체(요약 필요)
            paper_d = st.get("paper")
            if paper_d:
                p = Paper.from_dict(paper_d)
                p.track, p.slot = track, kind
                to_summarize.append(p)
                print(f"  [new]  {slot_id}: 새 선정작 → 재요약 대상")
            else:
                print(f"  [new]  {slot_id}: 선정작 없음(empty) → 슬롯 비움")
            # empty면 final에 넣지 않음 = 해당 슬롯 비움
        else:
            # 이번에 안 돌린 슬롯 → 기존 커밋 결과 유지
            keep = base_by_slot.get((track, kind))
            if keep is not None:
                final_by_slot[(track, kind)] = keep
                print(f"  [keep] {slot_id}: 기존 요약 유지")
            else:
                print(f"  [skip] {slot_id}: staging·기존 모두 없음")

    # 새로 staged된 슬롯만 요약 (호출 최소화)
    if to_summarize:
        n_lines = int(cfg.get("llm.summary_lines", 5))
        if args.dry_run:
            for p in to_summarize:
                _, summary, tags = _placeholder_summary(p, n_lines)
                d = p.to_dict()
                d["title_ko"], d["summary_ko"], d["tags"] = "", summary, tags
                final_by_slot[(p.track, p.slot)] = d
        else:
            from .summarize import summarize_all  # noqa: PLC0415

            provider = _build_provider_verbose(cfg)
            print(f"\n새로 staged된 {len(to_summarize)}편 요약 시작...")
            for d in summarize_all(cfg, provider, to_summarize):
                final_by_slot[(d.get("track", ""), d.get("slot", ""))] = d
    else:
        print("  새로 staged된 슬롯 없음 — 기존 결과로 렌더만 갱신")

    items = [final_by_slot[k] for k in SLOT_ORDER if k in final_by_slot]
    if not items:
        print("\n선정된 논문이 없습니다.")
        return 1

    fp = save_week_data(week_id, _make_payload(cfg, week_id, items))
    print(f"  데이터 저장: {fp}  ({len(items)}슬롯)")
    render_site(cfg)
    print("\n완료.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="주간 논문 브리핑 빌더")
    parser.add_argument("--config", default=None, help="config.yaml 경로")
    parser.add_argument("--week-id", default=None, help="주차 ID(YYYY-MM-DD) 강제 지정")
    parser.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 실행")
    parser.add_argument("--no-citations", action="store_true", help="Semantic Scholar 호출 생략")
    parser.add_argument("--render-only", action="store_true", help="수집/요약 없이 기존 data/로 사이트만 재생성")
    parser.add_argument(
        "--slot", default=None,
        help=f"단일 슬롯만 실행 후 staging 저장 ({' | '.join(SLOT_IDS)})",
    )
    parser.add_argument("--assemble", action="store_true", help="staging+기존 결과 조립 → 요약 → 저장 → 렌더")
    parser.add_argument("--print-week-id", action="store_true", help="현재 주차 ID만 출력하고 종료")
    args = parser.parse_args(argv)

    if args.print_week_id:
        cfg = Config.load(args.config)
        print(args.week_id or compute_week_id(cfg.get("site.timezone", "Asia/Seoul")))
        return 0
    if args.render_only:
        render_site(Config.load(args.config))
        return 0
    if args.assemble:
        return run_assemble(args)
    if args.slot:
        return run_single_slot(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
