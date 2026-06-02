"""arXiv 논문 수집 + Semantic Scholar 인용수·저자 소속 조회 + Hugging Face Papers 트렌딩."""
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import feedparser
import requests

from .config import Config, env

ARXIV_API = "https://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"
HF_DAILY_PAPERS = "https://huggingface.co/api/daily_papers"
USER_AGENT = "ResearchProject/1.0 (https://github.com/solsolv/weekly-paper-briefing-ai)"

# arXiv 가이드: 모든 호출 사이 최소 3~5초 간격. 4초 채택 (안전 마진 포함).
_ARXIV_MIN_INTERVAL_SEC = 4.0
_LAST_ARXIV_CALL_TS: float = 0.0


@dataclass
class Paper:
    arxiv_id: str                 # 버전 제거된 ID (예: "2401.01234")
    title: str
    authors: list[str]
    summary: str                  # arXiv 초록
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    abs_url: str
    track: str = ""               # "ai" | "mech"
    slot: str = ""                # "impact" | "latest" (선정 슬롯)
    citation_count: int | None = None
    influential_citation_count: int | None = None
    s2_url: str | None = None
    tldr: str | None = None       # Semantic Scholar TLDR (있으면)
    matched_keywords: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)   # 모든 고유 소속 기관
    source: str = "arxiv"         # "arxiv" | "huggingface"
    hf_upvotes: int | None = None

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "categories": self.categories,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "pdf_url": self.pdf_url,
            "abs_url": self.abs_url,
            "track": self.track,
            "slot": self.slot,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "s2_url": self.s2_url,
            "tldr": self.tldr,
            "matched_keywords": self.matched_keywords,
            "affiliations": self.affiliations,
            "source": self.source,
            "hf_upvotes": self.hf_upvotes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        """to_dict()로 직렬화한 dict에서 Paper 복원 (슬롯 staging 복구용).

        published/updated는 isoformat 문자열을 datetime으로 되돌린다.
        요약 단계에서 부가된 title_ko/summary_ko/tags 등 Paper에 없는 키는 무시한다.
        """
        def _dt(v):
            if isinstance(v, datetime):
                return v
            return datetime.fromisoformat(v) if v else datetime.now(timezone.utc)

        return cls(
            arxiv_id=d.get("arxiv_id", ""),
            title=d.get("title", ""),
            authors=list(d.get("authors", []) or []),
            summary=d.get("summary", ""),
            categories=list(d.get("categories", []) or []),
            published=_dt(d.get("published")),
            updated=_dt(d.get("updated")),
            pdf_url=d.get("pdf_url", ""),
            abs_url=d.get("abs_url", ""),
            track=d.get("track", ""),
            slot=d.get("slot", ""),
            citation_count=d.get("citation_count"),
            influential_citation_count=d.get("influential_citation_count"),
            s2_url=d.get("s2_url"),
            tldr=d.get("tldr"),
            matched_keywords=list(d.get("matched_keywords", []) or []),
            affiliations=list(d.get("affiliations", []) or []),
            source=d.get("source", "arxiv"),
            hf_upvotes=d.get("hf_upvotes"),
        )


def _parse_arxiv_datetime(value: str) -> datetime:
    # arXiv: "2024-01-15T12:34:56Z"
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _strip_version(arxiv_id_url: str) -> str:
    # "http://arxiv.org/abs/2401.01234v2" -> "2401.01234"
    tail = arxiv_id_url.rstrip("/").split("/")[-1]
    if "v" in tail:
        head, _, ver = tail.rpartition("v")
        if ver.isdigit():
            return head
    return tail


def _build_query(categories: list[str], keywords: list[str]) -> str:
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cat_clause})"
    if keywords:
        kw_clause = " OR ".join(f'abs:"{k}"' for k in keywords)
        query = f"{query} AND ({kw_clause})"
    return query


def _arxiv_entries_to_papers(entries) -> list[Paper]:
    papers: list[Paper] = []
    for entry in entries:
        try:
            arxiv_id = _strip_version(entry.id)
            published = _parse_arxiv_datetime(entry.published)
            updated = _parse_arxiv_datetime(entry.updated)
            cats = [t["term"] for t in getattr(entry, "tags", [])] or [getattr(entry, "arxiv_primary_category", {}).get("term", "")]
            pdf_url = ""
            for link in getattr(entry, "links", []):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
            papers.append(
                Paper(
                    arxiv_id=arxiv_id,
                    title=" ".join(entry.title.split()),
                    authors=[a.name for a in getattr(entry, "authors", [])],
                    summary=" ".join(entry.summary.split()),
                    categories=[c for c in cats if c],
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                    abs_url=f"https://arxiv.org/abs/{arxiv_id}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - 개별 엔트리 파싱 실패는 건너뜀
            print(f"  [warn] arXiv 엔트리 파싱 실패: {exc}")
    return papers


def _arxiv_get(params: dict) -> requests.Response:
    """arXiv API GET + 호출 간 최소 간격 보장 + 429/5xx 백오프 재시도.

    가이드: 모든 호출 사이 최소 3~5초. 호출 직전 마지막 호출 이후 경과시간을 확인,
    부족하면 sleep으로 보장한다.
    """
    global _LAST_ARXIV_CALL_TS
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    resp = None
    waits = [10, 30, 60, 120, 240]
    for attempt, wait in enumerate(waits):
        elapsed = time.time() - _LAST_ARXIV_CALL_TS
        if elapsed < _ARXIV_MIN_INTERVAL_SEC:
            time.sleep(_ARXIV_MIN_INTERVAL_SEC - elapsed)
        _LAST_ARXIV_CALL_TS = time.time()
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            print(f"  [warn] arXiv {resp.status_code}, {wait}s 후 재시도 ({attempt+1}/{len(waits)})")
            time.sleep(wait)
            continue
        break
    resp.raise_for_status()
    return resp


def fetch_arxiv(categories: list[str], keywords: list[str], max_results: int) -> list[Paper]:
    """arXiv API에서 카테고리/키워드에 맞는 최신 논문을 가져온다 (제출일 내림차순)."""
    search_query = _build_query(categories, keywords)
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = _arxiv_get(params)
    feed = feedparser.parse(resp.text)
    return _arxiv_entries_to_papers(feed.entries)


def fetch_arxiv_by_ids(arxiv_ids: list[str]) -> list[Paper]:
    """arXiv ID 리스트로 본문 메타데이터 조회 (HF Papers 후보 풀 보강용)."""
    if not arxiv_ids:
        return []
    params = {
        "id_list": ",".join(arxiv_ids),
        "max_results": len(arxiv_ids),
    }
    resp = _arxiv_get(params)
    feed = feedparser.parse(resp.text)
    return _arxiv_entries_to_papers(feed.entries)


def _within_lookback(paper: Paper, lookback_days: int, now: datetime) -> bool:
    cutoff = now - timedelta(days=lookback_days)
    return paper.published >= cutoff or paper.updated >= cutoff


def _matched_keywords(paper: Paper, keywords: list[str]) -> list[str]:
    text = f"{paper.title}\n{paper.summary}".lower()
    return [k for k in keywords if k.lower() in text]


def _normalize_title_key(title: str) -> str:
    """dedupe용 제목 정규화 — 소문자 + 알파넘만 유지."""
    import re
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _parse_s2_date(value: str | None) -> datetime | None:
    """S2 publicationDate('YYYY-MM-DD') 또는 year만 있을 때 처리."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def fetch_s2_search(cfg: Config, lookback_days: int, sort: str) -> list[Paper]:
    """Semantic Scholar /paper/search/bulk 로 후보 풀 보강.

    arxiv에 없는 저널·컨퍼런스 논문(예: JFM, IJSS, ASME)을 흡수.
    응답 abstract가 null이면 제외(요약 불가).
    """
    if not cfg.get("s2_search.enabled", False):
        return []
    query = cfg.get("s2_search.query", "") or ""
    if not query.strip():
        return []
    fields_of_study = cfg.get("s2_search.fields_of_study", []) or []
    limit = int(cfg.get("s2_search.limit", 100))
    # year 범위: 올해 - (lookback_days // 365 + 1)년 ~ 올해+1
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    year_range = f"{cutoff.year}-{now.year + 1}"

    api_key = env(cfg.get("semantic_scholar.api_key_env", "SEMANTIC_SCHOLAR_API_KEY"))
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "year": year_range,
        "sort": sort,
        "limit": min(limit, 1000),
        # bulk search는 authors.affiliations 포함 시 즉시 429/400 (nested expansion 부담).
        # affiliations 미요청 → 후속 enrich_citations 단계의 단건 GET이 affiliations 채움.
        "fields": "title,abstract,year,publicationDate,externalIds,citationCount,influentialCitationCount,authors.name,openAccessPdf,url",
    }
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    url = f"{S2_API}/search/bulk"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 429:
            print(f"  [warn] S2 search 429 — 빈 결과 반환")
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] S2 search 실패: {exc}")
        return []

    items = data.get("data") or []
    out: list[Paper] = []
    for item in items:
        abstract = (item.get("abstract") or "").strip()
        if not abstract:
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        pub = _parse_s2_date(item.get("publicationDate"))
        if pub is None:
            year = item.get("year")
            if not year:
                continue
            pub = datetime(int(year), 1, 1, tzinfo=timezone.utc)
        ext = item.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv") or ""
        doi = ext.get("DOI") or ""
        # dedupe key 우선순위: arxiv_id > "doi:..." > "s2:..."
        if arxiv_id:
            pid = arxiv_id
        elif doi:
            pid = f"doi:{doi}"
        else:
            pid = f"s2:{item.get('paperId', '')}"
        pdf_url = ""
        oa = item.get("openAccessPdf") or {}
        if isinstance(oa, dict):
            pdf_url = oa.get("url") or ""
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        affs: list[str] = []
        for a in (item.get("authors") or []):
            for af in (a.get("affiliations") or []):
                if af and af not in affs:
                    affs.append(af)
        p = Paper(
            arxiv_id=pid,
            title=" ".join(title.split()),
            authors=authors,
            summary=" ".join(abstract.split()),
            categories=[],
            published=pub,
            updated=pub,
            pdf_url=pdf_url,
            abs_url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
            citation_count=item.get("citationCount"),
            influential_citation_count=item.get("influentialCitationCount"),
            s2_url=item.get("url"),
            affiliations=affs,
            source="s2_search",
        )
        out.append(p)
    return out


def collect_candidates(
    cfg: Config,
    track: str,
    lookback_days: int,
    now: datetime | None = None,
    s2_sort: str | None = None,
) -> list[Paper]:
    """한 트랙(ai|mech)의 후보 논문 풀을 수집·필터링하여 반환.
    lookback_days는 호출 측에서 슬롯별로 지정 (임팩트는 365, 최신은 14 등).
    s2_sort: S2 Search 정렬 키 ('citationCount:desc' | 'publicationDate:desc'). None이면 S2 보강 생략.
    """
    now = now or datetime.now(timezone.utc)
    track_cfg = cfg.get(f"arxiv.tracks.{track}", {})
    categories = track_cfg.get("categories", [])
    keywords = track_cfg.get("keywords", []) or []
    keywords_domain = track_cfg.get("keywords_domain", []) or []
    max_results = cfg.get("arxiv.max_results_per_query", 400)
    min_abs = cfg.get("selection.min_abstract_chars", 0)

    raw = fetch_arxiv(categories, keywords, max_results)
    # S2 Search 보강 (트랙이 apply_to_tracks에 포함될 때만)
    s2_tracks = cfg.get("s2_search.apply_to_tracks", []) or []
    if s2_sort and track in s2_tracks:
        s2_raw = fetch_s2_search(cfg, lookback_days, s2_sort)
        if s2_raw:
            print(f"  S2 Search 보강: {len(s2_raw)}편 추가")
        raw = raw + s2_raw

    out: list[Paper] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    for p in raw:
        if p.arxiv_id in seen_ids:
            continue
        tkey = _normalize_title_key(p.title)
        if tkey and tkey in seen_titles:
            continue
        if not _within_lookback(p, lookback_days, now):
            continue
        if len(p.summary) < min_abs:
            continue
        if keywords:
            mk = _matched_keywords(p, keywords)
            if not mk:
                continue
            p.matched_keywords = mk
        if keywords_domain:
            mk_dom = _matched_keywords(p, keywords_domain)
            if not mk_dom:
                continue
            p.matched_keywords = p.matched_keywords + mk_dom
        p.track = track
        seen_ids.add(p.arxiv_id)
        if tkey:
            seen_titles.add(tkey)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# Semantic Scholar 인용수 + 저자 소속 기관
# --------------------------------------------------------------------------
def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def enrich_citations(cfg: Config, papers: list[Paper]) -> None:
    """papers의 citation_count / tldr / 저자 소속 기관을 Semantic Scholar로 채운다 (in-place)."""
    if not cfg.get("semantic_scholar.enabled", True):
        for p in papers:
            p.citation_count = 0
        return

    api_key = env(cfg.get("semantic_scholar.api_key_env", "SEMANTIC_SCHOLAR_API_KEY"))
    delay = float(cfg.get("semantic_scholar.request_delay_sec", 1.1))
    max_retries = int(cfg.get("semantic_scholar.max_retries", 3))
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
        delay = max(delay, 0.05)
    # authors.affiliations까지 한 번에 가져옴 (추가 API 호출 없음)
    fields = "citationCount,influentialCitationCount,url,tldr,title,authors.name,authors.affiliations"

    for p in papers:
        # S2 Search로 가져온 논문은 이미 cc/infl/affiliations 채워짐 — 재조회 불필요
        if p.source == "s2_search":
            if p.citation_count is None:
                p.citation_count = 0
            continue
        url = f"{S2_API}/arXiv:{p.arxiv_id}?fields={fields}"
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 404:
                    p.citation_count = 0
                    break
                if resp.status_code == 429:
                    wait = delay * (attempt + 2)
                    print(f"  [warn] S2 rate limit, {wait:.1f}s 대기")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                p.citation_count = data.get("citationCount") or 0
                p.influential_citation_count = data.get("influentialCitationCount") or 0
                p.s2_url = data.get("url")
                tldr = data.get("tldr")
                if isinstance(tldr, dict):
                    p.tldr = tldr.get("text")
                # 저자 소속 기관 (모든 저자에서 모은 뒤 중복 제거)
                affs: list[str] = []
                for a in (data.get("authors") or []):
                    for aff in (a.get("affiliations") or []):
                        if isinstance(aff, str):
                            affs.append(aff.strip())
                p.affiliations = _dedupe_preserve_order(affs)
                break
            except requests.RequestException as exc:
                if attempt == max_retries - 1:
                    print(f"  [warn] S2 조회 실패({p.arxiv_id}): {exc}")
                    p.citation_count = p.citation_count or 0
                else:
                    time.sleep(delay)
        time.sleep(delay)


# --------------------------------------------------------------------------
# Hugging Face Papers (AI 최신 슬롯 후보 풀)
# --------------------------------------------------------------------------
def fetch_huggingface_weekly(top_n: int = 10, days_back: int = 7) -> list[dict]:
    """HF Daily Papers를 최근 days_back일 모아 upvote 내림차순 상위 top_n개 반환.
    반환 형식: [{"arxiv_id": "2401.01234", "upvotes": 123, "title": "..."}]
    실패/빈 결과 시 빈 리스트."""
    items: dict[str, dict] = {}
    today = datetime.now(timezone.utc).date()
    for d in range(days_back):
        date = today - timedelta(days=d)
        params = {"date": date.isoformat()}
        try:
            resp = requests.get(
                HF_DAILY_PAPERS,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=20,
            )
            if resp.status_code != 200:
                continue
            data = resp.json() or []
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] HF Papers 조회 실패 {date}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            paper = entry.get("paper") or entry  # 응답 형식이 약간 다를 수 있음
            aid = paper.get("id") or paper.get("arxivId") or paper.get("arxiv_id")
            if not aid:
                continue
            upv = paper.get("upvotes") or paper.get("upvotes_count") or 0
            title = paper.get("title") or entry.get("title") or ""
            prev = items.get(aid)
            if prev is None or upv > prev["upvotes"]:
                items[aid] = {"arxiv_id": str(aid).strip(), "upvotes": int(upv or 0), "title": title}
    if not items:
        return []
    sorted_items = sorted(items.values(), key=lambda x: x["upvotes"], reverse=True)
    return sorted_items[:top_n]
