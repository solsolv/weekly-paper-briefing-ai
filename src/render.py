"""주별 JSON 저장 + Jinja2로 docs/ 정적 사이트 생성."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import DATA_DIR, DOCS_DIR, TEMPLATES_DIR, Config

TRACK_LABEL = {"ai": "AI", "mech": "AI 활용 기계공학"}


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
        "interests": {
            "ai": cfg.get("interests.ai", []) or [],
            "mech": cfg.get("interests.mech", []) or [],
        },
    }
    weeks = load_all_weeks()

    # 주별 상세 페이지
    week_tmpl = env.get_template("week.html")
    for w in weeks:
        html = week_tmpl.render(site=site, week=w, track_label=TRACK_LABEL, now=datetime.now(timezone.utc))
        (weeks_dir / f"{w['week_id']}.html").write_text(html, encoding="utf-8")

    # 인덱스
    index_tmpl = env.get_template("index.html")
    html = index_tmpl.render(site=site, weeks=weeks, track_label=TRACK_LABEL, now=datetime.now(timezone.utc))
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"  사이트 생성 완료: {DOCS_DIR}  (주차 {len(weeks)}개)")
