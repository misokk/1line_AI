"""Create fictional Korean/English lesson pages for offline recognition tests.

The fixture filename is historical; no provider API is called.
"""

from pathlib import Path

import pymupdf


def create_pdf(path: Path) -> None:
    text = (
        "문장의 종류\n\n"
        "평서문: 사실이나 생각을 서술하는 문장입니다.\n"
        "예: She is a nurse.\n\n"
        "의문문: 상대방에게 질문하는 문장입니다.\n"
        "예: Is she a nurse?\n\n"
        "부정문: 사실이 아님을 나타내는 문장입니다.\n"
        "예: She is not a nurse.\n"
    )
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        remaining = page.insert_textbox(
            pymupdf.Rect(40, 40, 550, 780), text, fontname="korea", fontsize=14
        )
        if remaining < 0:
            raise ValueError("Korean fixture text does not fit on the page.")
        page = pdf.new_page()
        page.insert_text((40, 80), "이 페이지는 테스트 요청 대상이 아닙니다.", fontname="korea")
        pdf.save(path)


if __name__ == "__main__":
    create_pdf(Path(__file__).with_name("groq_korean.pdf"))
