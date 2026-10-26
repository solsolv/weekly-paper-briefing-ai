"""주간 검토 메일 — 토요일 빌드 끝에 본인 Gmail로 발송 (GitHub Actions + Gmail SMTP).

- 성공(success)     : 인사말 + 주차 페이지 상단 캡처 + 사이트 링크 (팀원에게 그대로 전달 가능한 문구)
- 보충 완료(refill) : 본문은 성공과 동일, 제목 말머리만 다름 (수동 재실행으로 빠진 슬롯을 채웠을 때)
- 미완료(incomplete): 실패·누락 슬롯 + 재실행 방법 + 링크 (캡처 없음)

계정·비밀번호·받는 주소는 코드에 두지 않고 환경변수(GitHub Secret)로만 받는다:
GMAIL_USER, GMAIL_APP_PASSWORD, MAIL_TO(콤마 구분 가능).
"""
from __future__ import annotations

import json
import os
import smtplib
import tempfile
from datetime import datetime
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import make_msgid
from html import escape
from pathlib import Path

from .config import DATA_DIR, DOCS_DIR, Config
from .og import _capture, find_browser

# 성공 판정에 쓰는 필수 슬롯 (국내 kr-latest는 0편이어도 정상 → 판정 제외)
REQUIRED_SLOTS = ["ai-impact", "ai-latest", "mech-impact", "mech-latest"]
_WEEKDAY_KO = "월화수목금토일"

# 메일용 캡처: 주차 페이지의 헤더 + '이번 주' 목록만. 공개 전 가림(JS 인라인 display:none)은
# !important로 덮어 해제하고, 배경 장식을 지워 흰 배경 기준으로 아래 여백을 잘라낸다.
_CAPTURE_CSS = """<style>
.bg-stage,.home-link,.filter-bar,.track-section,.back-pill-wrap,footer.site,#publish-notice,
[data-gate-content]>p{display:none!important}
[data-gate-content]{display:block!important}
.reveal{opacity:1!important;transform:none!important;transition:none!important}
html,body{background:#ffffff!important}
.wrap{padding-top:36px!important;padding-bottom:0!important}
header.site{margin-bottom:28px!important}
.toc{margin-bottom:0!important}
</style></head>"""


def _week_label(week_id: str) -> str:
    try:
        d = datetime.strptime(week_id, "%Y-%m-%d")
        return f"{week_id}({_WEEKDAY_KO[d.weekday()]})"
    except ValueError:
        return week_id


def capture_week_top(cfg: Config, week_id: str) -> bytes | None:
    """docs/weeks/<week>.html 상단(헤더+목록)을 넓은 폭으로 캡처해 PNG 바이트 반환. 실패 시 None."""
    page = DOCS_DIR / "weeks" / f"{week_id}.html"
    browser = find_browser()
    if not page.exists() or browser is None:
        print(f"  [warn] 캡처 생략 (페이지 {'있음' if page.exists() else '없음'}, 브라우저 {'있음' if browser else '없음'})")
        return None
    html = page.read_text(encoding="utf-8").replace("</head>", _CAPTURE_CSS, 1)
    width = int(cfg.get("notify.capture_width", 1000))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "capture.png"
        if not _capture(browser, html, out, width, int(cfg.get("notify.capture_max_height", 1400))):
            return None
        try:
            from PIL import Image, ImageChops  # noqa: PLC0415

            img = Image.open(out).convert("RGB")
            # 흰 배경과 다른 영역의 아래 끝 + 여백까지만 남김
            diff = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255)))
            bbox = diff.point(lambda v: 255 if v > 6 else 0).getbbox()
            if bbox:
                img = img.crop((0, 0, img.width, min(img.height, bbox[3] + 32)))
            img.save(out, optimize=True)
        except ImportError:
            print("  [warn] Pillow 없음 — 캡처 여백 자르기 생략")
        return out.read_bytes()


def _load_week(week_id: str) -> dict | None:
    fp = DATA_DIR / f"{week_id}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def missing_slots(week: dict | None) -> list[str]:
    """데이터 기준으로 비어 있는 필수 슬롯 (단계는 성공했지만 선정작이 없던 경우도 잡아냄)."""
    if not week:
        return list(REQUIRED_SLOTS)
    have = {f"{p.get('track')}-{p.get('slot')}" for p in week.get("papers", [])}
    return [s for s in REQUIRED_SLOTS if s not in have]


def _counts_sentence(week: dict) -> str:
    n = {t: len((week.get("tracks") or {}).get(t) or []) for t in ("ai", "mech", "kr")}
    base = f"AI {n['ai']}편, AI 활용 기계공학 {n['mech']}편"
    if n["kr"]:
        return f"이번 주에는 {base}과 함께 국내 학술지 논문 {n['kr']}편을 소개합니다."
    return f"이번 주에는 {base}을 소개합니다."


def _subject(cfg: Config, kind: str, week_id: str) -> str:
    prefix = {
        "success": cfg.get("notify.prefix_success", "[논문 브리핑]"),
        "refill": cfg.get("notify.prefix_refill", "[논문 브리핑 보충 완료]"),
        "incomplete": cfg.get("notify.prefix_incomplete", "[논문 브리핑 ⚠️ 미완료]"),
    }[kind]
    return f"{prefix} {cfg.get('site.title', '')} - {week_id} 주차"


def _briefing_body(cfg: Config, week: dict, week_id: str, has_image: bool) -> tuple[str, str]:
    """성공·보충 완료 공통 본문 → (text, html)."""
    site = (cfg.get("site.base_url", "") or "").rstrip("/") + "/"
    title = cfg.get("notify.briefing_name", "AI & AI 활용 기계공학 논문 브리핑")
    lines = [
        "안녕하세요.",
        f"{_week_label(week_id)} 주차 {title}입니다.",
        "",
        "매주 쏟아지는 AI와 AI 활용 기계공학 분야의 논문 가운데, 영향력 있는 논문과 최신 논문을 골라 "
        "한국어 요약으로 정리하고 있습니다.",
        _counts_sentence(week),
        "업무 시 참고 부탁 드립니다.",
    ]
    tail = [
        f"브리핑 사이트: {site}",
        "(지난 주차 논문도 주제·키워드로 검색해 보실 수 있습니다.)",
        "",
        "다음 주에도 새로운 논문 소식으로 찾아뵙겠습니다.",
        "감사합니다.",
    ]
    text = "\n".join(lines + [""] + tail)

    def p(ls):
        return "<br>".join(escape(x) for x in ls)

    img = (f'<a href="{site}"><img src="cid:{{CID}}" alt="{escape(week_id)} 주차 논문 목록" '
           f'style="display:block;width:100%;max-width:640px;margin:16px 0;border:1px solid #e5e7eb"></a>'
           if has_image else "<br><br>")
    html = (
        f'<div style="font-family:\'Malgun Gothic\',\'Apple SD Gothic Neo\',sans-serif;font-size:14px;'
        f'line-height:1.8;color:#1d1d1f;max-width:680px">'
        f"{p(lines[:2])}<br><br>{p(lines[3:])}{img}"
        f'브리핑 사이트: <a href="{site}" style="color:#1a73e8">{escape(site)}</a><br>'
        f"{p(tail[1:])}</div>"
    )
    return text, html


def _incomplete_body(cfg: Config, week_id: str, failed: list[str], ok: list[str], run_url: str) -> tuple[str, str]:
    repo = cfg.get("notify.repo_url", "https://github.com/solsolv/weekly-paper-briefing-ai").rstrip("/")
    wf = f"{repo}/actions/workflows/weekly-briefing.yml"
    # 재실행 대상: 실패한 슬롯만. 조립·커밋 단계만 실패했다면 전체(all) 재실행 안내.
    rerun = [s for s in failed if s in REQUIRED_SLOTS] or ["all"]
    lines = [
        f"{week_id} 주차 논문 브리핑 빌드가 완료되지 않았습니다.",
        "",
        f"실패한 단계: {', '.join(failed)}",
        f"성공한 슬롯: {', '.join(ok) if ok else '없음'}",
        "",
        "Actions에서 Run workflow를 실행해 실패한 슬롯만 다시 채울 수 있습니다.",
        f"(slots: {','.join(rerun)} / week_id: {week_id})",
        "완료되면 '보충 완료' 메일이 다시 발송됩니다.",
        "",
        f"빌드 로그: {run_url or '-'}",
        f"실행 페이지: {wf}",
    ]
    text = "\n".join(lines)
    html_lines = []
    for ln in lines:
        e = escape(ln)
        for url in (run_url, wf):
            if url and url in ln:
                e = e.replace(escape(url), f'<a href="{escape(url)}" style="color:#1a73e8">{escape(url)}</a>')
        html_lines.append(e)
    html = (f'<div style="font-family:\'Malgun Gothic\',sans-serif;font-size:14px;line-height:1.8;color:#1d1d1f">'
            f'{"<br>".join(html_lines)}</div>')
    return text, html


def build_message(cfg: Config, week_id: str, *, failed: list[str], refill: bool, run_url: str = "") -> EmailMessage:
    week = _load_week(week_id)
    failed = sorted(set(failed) | set(missing_slots(week)), key=lambda s: (s not in REQUIRED_SLOTS, s))
    kind = "incomplete" if failed else ("refill" if refill else "success")

    msg = EmailMessage()
    msg["Subject"] = _subject(cfg, kind, week_id)
    sender = os.environ.get("GMAIL_USER", "sender@example.com")
    user, _, domain = sender.partition("@")
    msg["From"] = Address(cfg.get("notify.from_name", "주간 논문 브리핑"), user, domain or "example.com")
    msg["To"] = os.environ.get("MAIL_TO", "me@example.com")

    if kind == "incomplete":
        ok = [s for s in REQUIRED_SLOTS + ["kr-latest"] if s not in failed]
        text, html = _incomplete_body(cfg, week_id, failed, ok, run_url)
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
        return msg

    png = capture_week_top(cfg, week_id)
    text, html = _briefing_body(cfg, week, week_id, has_image=png is not None)
    msg.set_content(text)
    if png:
        cid = make_msgid(domain="briefing")
        msg.add_alternative(html.replace("{CID}", cid[1:-1]), subtype="html")
        # multipart/related로 본문 이미지 첨부 → 외부 이미지 차단과 무관하게 표시
        msg.get_payload()[1].add_related(png, maintype="image", subtype="png", cid=cid,
                                         filename=f"briefing-{week_id}.png")
    else:
        msg.add_alternative(html, subtype="html")
    return msg


def send(msg: EmailMessage) -> None:
    user = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to = [a.strip() for a in os.environ.get("MAIL_TO", "").split(",") if a.strip()]
    if not (user and password and to):
        raise RuntimeError("GMAIL_USER / GMAIL_APP_PASSWORD / MAIL_TO 환경변수(Secret)가 필요합니다")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as server:
        server.login(user, password.replace(" ", ""))
        server.send_message(msg, from_addr=user, to_addrs=to)


def run_notify(cfg: Config, week_id: str, *, failed: list[str], refill: bool, run_url: str,
               preview: str | None = None) -> int:
    if not cfg.get("notify.enabled", True):
        print("메일 발송 비활성 (notify.enabled=false)")
        return 0
    msg = build_message(cfg, week_id, failed=failed, refill=refill, run_url=run_url)
    print(f"  메일 제목: {msg['Subject']}")
    if preview:
        Path(preview).write_bytes(bytes(msg))   # .eml — 메일 앱으로 열어 확인 (발송 안 함)
        print(f"  미리보기 저장: {preview}")
        return 0
    send(msg)
    print("  메일 발송 완료")
    return 0
