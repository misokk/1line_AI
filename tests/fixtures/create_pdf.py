"""Generate the fictional, redistributable English grammar fixture without an API."""

from pathlib import Path

import pymupdf


def create_pdf(path: Path) -> None:
    texts = [
        "Unit 1: Simple Present\n\n"
        "The simple present describes habits and general truths.\n"
        "Use the base verb with I, you, we, and they.\n"
        "Add -s or -es with he, she, and it.\n"
        "Example: She studies English every day.\n"
        "Use do not or does not with the base verb for negatives.\n",
        "Unit 2: Present Continuous\n\n"
        "The present continuous describes actions happening now.\n"
        "Form it with am, is, or are followed by a verb ending in -ing.\n"
        "Example: They are studying English now.\n"
        "The simple present is for habits; the present continuous is for current actions.\n",
    ]
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 780), text, fontsize=12)
        doc.save(path)


if __name__ == "__main__":
    create_pdf(Path(__file__).with_name("english_grammar.pdf"))
