"""링크 미리보기(OG) 이미지·홈 화면 아이콘 생성 — headless Chrome/Edge로 HTML을 PNG로 캡처.

- 사이트 공통 이미지: docs/og/site.<hash>.png
- 주차별 이미지   : docs/og/<week_id>.<hash>.png  (그 주 논문 제목 목록)
- 홈 화면 아이콘  : docs/apple-touch-icon.png (iOS는 SVG 파비콘을 쓰지 않음)

파일명의 hash는 캡처할 HTML 내용의 해시 → 제목·템플릿이 바뀔 때만 다시 캡처하고, 메신저의
미리보기 캐시도 새 파일명으로 자연스럽게 갱신된다. 브라우저가 없으면 이미지 없이 진행한다
(페이지는 정상 생성, og:image만 생략).
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

OG_W, OG_H = 1200, 630
ICON_SIZE = 180

_BROWSER_CANDIDATES = [
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge",
]
_WINDOWS_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str | None:
    """CHROME_PATH 환경변수 → PATH 상의 Chrome/Chromium/Edge → Windows 기본 설치 경로 순."""
    env = os.environ.get("CHROME_PATH")
    if env and Path(env).exists():
        return env
    for name in _BROWSER_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    for p in _WINDOWS_PATHS:
        if Path(p).exists():
            return p
    return None


def _capture(browser: str, html: str, out: Path, width: int, height: int) -> bool:
    """HTML 문자열을 임시 파일로 저장 후 headless 캡처. 성공 여부 반환.

    Windows Edge는 캡처 후 종료 시 비정상 종료 코드를 내기도 해서, 종료 코드가 아니라
    결과 파일 존재 여부로 성공을 판단한다.
    """
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "page.html"
        src.write_text(html, encoding="utf-8")
        tmp_png = Path(td) / "shot.png"
        cmd = [
            browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--window-size={width},{height}", "--force-device-scale-factor=1",
            "--virtual-time-budget=6000",            # 웹폰트 로딩 대기
            f"--screenshot={tmp_png}", src.as_uri(),
        ]
        if platform.system() == "Linux":
            cmd.insert(1, "--no-sandbox")            # CI 컨테이너 환경 대비
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  [warn] OG 캡처 실패({out.name}): {exc}")
            return False
        if not tmp_png.exists() or tmp_png.stat().st_size == 0:
            print(f"  [warn] OG 캡처 결과 없음({out.name})")
            return False
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_png), out)
        return True


def _hashed_name(prefix: str, html: str) -> str:
    return f"{prefix}.{hashlib.sha1(html.encode('utf-8')).hexdigest()[:10]}.png"


def _ensure(browser: str | None, og_dir: Path, prefix: str, html: str, w: int, h: int) -> str | None:
    """prefix.<hash>.png가 있으면 재사용, 없으면 캡처 후 같은 prefix의 옛 파일 삭제. 파일명 반환."""
    name = _hashed_name(prefix, html)
    target = og_dir / name
    if target.exists():
        return name
    if browser is None or not _capture(browser, html, target, w, h):
        return None
    for old in og_dir.glob(f"{prefix}.*.png"):
        if old.name != name:
            old.unlink()
    print(f"  OG 이미지 생성: og/{name}")
    return name


def build_og_images(env, docs_dir: Path, static_dir: Path, site: dict, weeks: list[dict], track_label: dict) -> dict:
    """OG 이미지·아이콘 생성. 반환: {"site": 파일명|None, "weeks": {week_id: 파일명}, "touch_icon": bool}."""
    browser = find_browser()
    if browser is None:
        print("  [warn] headless 브라우저(Chrome/Edge)를 찾지 못해 OG 이미지 생성을 건너뜀")
    og_dir = docs_dir / "og"
    fav = (static_dir / "favicon.svg").read_text(encoding="utf-8")
    common = {"w": OG_W, "h": OG_H, "favicon_svg": fav, "publish_hour": site.get("publish_hour", 8)}

    result: dict = {"site": None, "weeks": {}, "touch_icon": False}
    html = env.get_template("og/site.html").render(**common)
    result["site"] = _ensure(browser, og_dir, "site", html, OG_W, OG_H)

    week_tmpl = env.get_template("og/week.html")
    for w in weeks:
        html = week_tmpl.render(week=w, track_label=track_label, **common)
        name = _ensure(browser, og_dir, w["week_id"], html, OG_W, OG_H)
        if name:
            result["weeks"][w["week_id"]] = name

    icon = docs_dir / "apple-touch-icon.png"
    icon_html = env.get_template("og/icon.html").render(w=ICON_SIZE, h=ICON_SIZE, favicon_svg=fav)
    stamp = docs_dir / "og" / ".touch-icon.hash"
    digest = hashlib.sha1(icon_html.encode("utf-8")).hexdigest()
    if icon.exists() and stamp.exists() and stamp.read_text(encoding="utf-8").strip() == digest:
        result["touch_icon"] = True
    elif browser is not None and _capture(browser, icon_html, icon, ICON_SIZE, ICON_SIZE):
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(digest, encoding="utf-8")
        result["touch_icon"] = True
    else:
        result["touch_icon"] = icon.exists()
    return result
