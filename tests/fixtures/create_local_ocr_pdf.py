"""Create known native, scanned and incomplete-text Korean/English/math pages."""

import os
from pathlib import Path

import pymupdf

TEXT = (
    "문장의 종류와 기초 수학\n\n"
    "평서문은 사실을 설명하는 문장입니다.\n"
    "She is a nurse.\n\n"
    "의문문은 상대방에게 질문하는 문장입니다.\n"
    "Is she a nurse?\n\n"
    "부정문은 사실이 아님을 나타냅니다.\n"
    "She is not a nurse.\n\n"
    "Equation: x + 2 = 5\n"
    "Answer: x = 3\n"
)


def create_pdf(path: Path) -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        font = Path(os.environ.get("WINDIR", "")) / "Fonts" / "malgun.ttf"
        if (
            page.insert_textbox(
                pymupdf.Rect(50, 50, 545, 790),
                TEXT,
                fontname="fixture" if font.is_file() else "korea",
                fontfile=str(font) if font.is_file() else None,
                fontsize=17,
            )
            < 0
        ):
            raise ValueError("Fixture does not fit.")
        image = page.get_pixmap(dpi=200).tobytes("png")
        page = pdf.new_page()
        page.insert_image(page.rect, stream=image)
        page = pdf.new_page()
        page.insert_text((50, 60), "She is a nurse. Is she a nurse? She is not a nurse.")
        page.insert_image(page.rect, stream=image)
        # A shorter, clean scan checks the successful OCR path alongside harder prose.
        with pymupdf.open() as simple:
            source = simple.new_page()
            source.insert_text((50, 80), "기초 수학", fontname="korea", fontsize=22)
            source.insert_text((50, 130), "English and Math", fontsize=18)
            source.insert_text((50, 180), "Equation: x + 2 = 5", fontsize=18)
            source.insert_text((50, 230), "Answer: x = 3", fontsize=18)
            scan = source.get_pixmap(dpi=200).tobytes("png")
        page = pdf.new_page()
        page.insert_image(page.rect, stream=scan)
        pdf.subset_fonts()
        pdf.save(path, deflate=True, garbage=4)


if __name__ == "__main__":
    create_pdf(Path(__file__).with_name("local_ocr.pdf"))
