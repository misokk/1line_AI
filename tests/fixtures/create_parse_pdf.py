"""Fictional mixed-language/table/equation/chart fixture; no user documents."""

import os
from pathlib import Path

import pymupdf


def create_pdf(path: Path):
    font = Path(os.environ.get("WINDIR", "")) / "Fonts" / "malgun.ttf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        name = "fixture" if font.is_file() else "korea"
        page.insert_font(fontname=name, fontfile=str(font) if font.is_file() else None)
        page.insert_text((50, 65), "한국어 영어 수학 학습 자료", fontname=name, fontsize=20)
        page.insert_text(
            (50, 105), "평서문: 사실을 설명하는 문장입니다.", fontname=name, fontsize=14
        )
        page.insert_text((50, 135), "She is a nurse. Is she a nurse?", fontsize=14)
        page.insert_text((50, 175), "Equation: x + 2 = 5; Answer: x = 3", fontsize=14)
        page.insert_text((50, 215), "표 1. 과목별 문제 수", fontname=name, fontsize=14)
        for y in (235, 275, 315, 355):
            page.draw_line((50, y), (450, y))
        for x in (50, 250, 450):
            page.draw_line((x, 235), (x, 355))
        for x, y, text in (
            (65, 260, "과목"),
            (265, 260, "문제 수"),
            (65, 300, "영어"),
            (265, 300, "3"),
            (65, 340, "수학"),
            (265, 340, "2"),
        ):
            page.insert_text((x, y), text, fontname=name, fontsize=14)
        page = pdf.new_page()
        page.insert_font(fontname=name, fontfile=str(font) if font.is_file() else None)
        page.insert_text((50, 65), "학습 진도 그래프", fontname=name, fontsize=20)
        page.insert_text((50, 110), "월요일 20, 화요일 40, 수요일 60", fontname=name, fontsize=14)
        page.draw_line((80, 380), (460, 380))
        page.draw_line((80, 380), (80, 180))
        page.draw_polyline([(130, 330), (260, 280), (390, 230)], color=(0, 0, 1), width=3)
        for x, y, label in ((130, 330, "20"), (260, 280, "40"), (390, 230, "60")):
            page.draw_circle((x, y), 5, fill=(0, 0, 1))
            page.insert_text((x, y - 12), label, fontsize=14)
        page.insert_text((95, 410), "Mon                 Tue                 Wed", fontsize=14)
        pdf.subset_fonts()
        pdf.save(path, garbage=4, deflate=True)


if __name__ == "__main__":
    create_pdf(Path(__file__).with_name("document_parse.pdf"))
