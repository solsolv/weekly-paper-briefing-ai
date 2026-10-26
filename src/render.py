"""주별 JSON 저장 + Jinja2로 docs/ 정적 사이트 생성."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import DATA_DIR, DOCS_DIR, TEMPLATES_DIR, Config

TRACK_LABEL = {"ai": "AI", "mech": "AI 활용 기계공학", "kr": "국내 논문"}
STATIC_DIR = TEMPLATES_DIR.parent / "static"   # favicon 등 그대로 복사할 파일


def save_week_data(week_id: str, payload: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / f"{week_id}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def load_all_weeks() -> list[dict]:
    weeks: list[dict] = []
    if not DATA_DIR.exists():
        return weeks
    for fp in sorted(DATA_DIR.glob("*.json"), reverse=True):
        try:
            weeks.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {fp.name} 로드 실패: {exc}")
    return weeks


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return iso[:10]


_WEEKDAY_KO = "월화수목금토일"


def _updated_label(tz_name: str) -> str:
    """빌드 시각을 사이트 타임존 날짜로 표기 (예: '2026-10-17 (토)')."""
    now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        now = now.astimezone(ZoneInfo(tz_name))
    except Exception:  # noqa: BLE001 - tzdata 미설치 등 → UTC 그대로
        pass
    return f"{now:%Y-%m-%d} ({_WEEKDAY_KO[now.weekday()]})"


def _search_items(weeks: list[dict]) -> str:
    """메인 페이지 검색·필터용 논문 목록(JSON 문자열). publish_at은 페이지 JS가 공개 전 주차를 숨기는 데 사용."""
    items = []
    for w in weeks:
        for p in w.get("papers", []):
            items.append({
                "w": w["week_id"], "pub": w.get("publish_at") or "", "id": p.get("arxiv_id", ""),
                "tr": p.get("track", ""), "sl": p.get("slot", ""),
                "tk": p.get("title_ko") or "", "te": p.get("title", ""),
                "au": ", ".join((p.get("authors") or [])[:4]),
                "s": (p.get("summary_ko") or "").replace("\n", " "),
                "tg": p.get("tags") or [], "tp": p.get("topics") or [], "v": p.get("venue") or "",
            })
    # </script> 조기 종료 방지
    return json.dumps(items, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _copy_static() -> None:
    if STATIC_DIR.exists():
        for fp in STATIC_DIR.iterdir():
            if fp.is_file():
                shutil.copy2(fp, DOCS_DIR / fp.name)


def render_site(cfg: Config) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    weeks_dir = DOCS_DIR / "weeks"
    weeks_dir.mkdir(parents=True, exist_ok=True)

    env = _env()
    env.filters["fmt_date"] = _fmt_date
    site = {
        "title": cfg.get("site.title", "주간 논문 브리핑"),
        "description": cfg.get("site.description", ""),
        "base_url": (cfg.get("site.base_url", "") or "").rstrip("/"),
        "publish_hour": int(cfg.get("site.publish_hour", 8)),
        "developer": cfg.get("site.developer", {}) or {},
        "data_sources": cfg.get("site.data_sources", "arXiv · Semantic Scholar · LLM 요약"),
        "updated": _updated_label(cfg.get("site.timezone", "Asia/Seoul")),
        "interests": {
            "ai": cfg.get("interests.ai", []) or [],
            "mech": cfg.get("interests.mech", []) or [],
        },
    }
    weeks = load_all_weeks()
    _copy_static()

    # 링크 미리보기 이미지 (브라우저 없으면 건너뜀 → og:image 생략)
    from .og import build_og_images  # noqa: PLC0415

    og = build_og_images(env, DOCS_DIR, STATIC_DIR, site, weeks, TRACK_LABEL)
    site["touch_icon"] = og["touch_icon"]
    site["og_image"] = f"og/{og['site']}" if og["site"] else ""

    # 주별 상세 페이지
    week_tmpl = env.get_template("week.html")
    for w in weeks:
        week_og = og["weeks"].get(w["week_id"])
        html = week_tmpl.render(
            site=site, week=w, track_label=TRACK_LABEL, now=datetime.now(timezone.utc),
            og_image=f"og/{week_og}" if week_og else site["og_image"],
        )
        (weeks_dir / f"{w['week_id']}.html").write_text(html, encoding="utf-8")

    # 인덱스
    index_tmpl = env.get_template("index.html")
    html = index_tmpl.render(
        site=site, weeks=weeks, track_label=TRACK_LABEL, now=datetime.now(timezone.utc),
        og_image=site["og_image"], search_json=_search_items(weeks),
    )
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"  사이트 생성 완료: {DOCS_DIR}  (주차 {len(weeks)}개)")
