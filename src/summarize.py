"""선정된 논문을 LLM으로 한국어 요약."""
from __future__ import annotations

from .collect import Paper
from .config import Config
from .providers import LLMProvider

SYSTEM_PROMPT = (
    "당신은 AI/머신러닝과 'AI를 활용한 기계공학' 연구를 깊이 이해하는 한국어 과학 리뷰어입니다. "
    "주어진 논문의 제목과 초록(필요시 TLDR)만으로, 비전문가도 핵심을 파악할 수 있도록 "
    "정확하고 군더더기 없는 한국어 요약을 작성합니다. 과장·추측은 피하고, 초록에 없는 수치는 지어내지 않습니다."
)

PROMPT_TEMPLATE = """다음 논문을 한국어로 요약하세요.

[제목] {title}
[저자] {authors}
[arXiv 분류] {categories}
[arXiv 초록]
{summary}
{tldr_block}

다음 형식으로 정확히 출력하세요. 머리말·맺음말·제목 반복 없이 아래 세 부분만 출력합니다.

TITLE_KO: <영문 제목을 자연스러운 한국어로 번역. 고유명·약어는 원어 유지(예: GPT, CFD, LLM). 한 줄.>

- 불릿1
- 불릿2
- 불릿3
- 불릿4
- 불릿5

TAGS: <쉼표로 구분한 정확히 {n_tags}개의 한국어 키워드>

요구사항:
- 불릿은 **정확히 {n_lines}개**. 각 불릿은 한 문장, 35~55자 내외.
- 5개 불릿의 흐름: ① 문제·배경 + 기존 한계 ② 제안 방법의 핵심 아이디어 ③ 작동 방식·실험 설정 ④ 주요 결과(초록에 수치가 있으면 포함) ⑤ 차별점·기여와 한 줄 총평.
- 전문 용어는 필요하면 영어를 괄호로 병기.
- 태그는 서로 다른 측면을 다루도록(예: 방법론·응용분야·핵심기술·도메인).
"""


def _format_tldr(paper: Paper) -> str:
    if paper.tldr:
        return f"[Semantic Scholar TLDR]\n{paper.tldr}\n"
    return ""


def _split_response(text: str) -> tuple[str, str, list[str]]:
    """LLM 응답을 (title_ko, summary_bullets, tags) 로 분리."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    title_ko = ""
    tags: list[str] = []
    body: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not title_ko and s.upper().startswith("TITLE_KO:"):
            title_ko = s.split(":", 1)[1].strip()
            continue
        if s.upper().startswith("TAGS:"):
            raw = s.split(":", 1)[1]
            tags = [t.strip() for t in raw.replace("，", ",").split(",") if t.strip()]
            continue
        body.append(ln)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return title_ko, "\n".join(body).strip(), tags


def summarize_paper(provider: LLMProvider, paper: Paper, n_lines: int, n_tags: int) -> tuple[str, str, list[str]]:
    prompt = PROMPT_TEMPLATE.format(
        title=paper.title,
        authors=", ".join(paper.authors[:8]) + (" 외" if len(paper.authors) > 8 else ""),
        categories=", ".join(paper.categories),
        summary=paper.summary,
        tldr_block=_format_tldr(paper),
        n_lines=n_lines,
        n_tags=n_tags,
    )
    raw = provider.generate(SYSTEM_PROMPT, prompt)
    return _split_response(raw)


def summarize_all(cfg: Config, provider: LLMProvider, papers: list[Paper]) -> list[dict]:
    """papers 각각을 요약하여 [{paper..., title_ko, summary_ko, tags}] 형태로 반환."""
    n_lines = int(cfg.get("llm.summary_lines", 5))
    n_tags = int(cfg.get("llm.tags_per_paper", 4))
    out: list[dict] = []
    for i, paper in enumerate(papers, 1):
        print(f"  [{i}/{len(papers)}] 요약 중: {paper.title[:60]}...")
        try:
            title_ko, summary, tags = summarize_paper(provider, paper, n_lines, n_tags)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] 요약 실패: {exc}")
            title_ko, summary, tags = "", "(요약 생성에 실패했습니다. 원문을 참고하세요.)", []
        item = paper.to_dict()
        # 국내 논문은 Crossref의 한글 원제목이 있으면 번역 대신 원제목 사용
        item["title_ko"] = paper.title_local or title_ko
        item["summary_ko"] = summary
        item["tags"] = tags
        out.append(item)
    return out
