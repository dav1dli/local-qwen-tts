import argparse
import re
from html.parser import HTMLParser
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ImportError:
    raise SystemExit("Install the Markdown parser with: python -m pip install markdown-it-py")


APPENDIX_HEADINGS = {
    "neue und schwierige wörter",
    "schwierige sätze",
    "wortliste",
    "wörterliste",
    "vokabeln",
    "vokabelliste",
    "word list",
    "word lists",
    "vocabulary",
    "glossary",
}


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag == "br" and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif tag in {"p", "div", "li"} and not self.hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def inline_text(children):
    parts = []
    hidden = 0
    for token in children or []:
        if token.type == "html_inline":
            if re.match(r"<\s*(script|style)\b", token.content, re.IGNORECASE):
                hidden += 1
            elif re.match(r"<\s*/\s*(script|style)\s*>", token.content, re.IGNORECASE):
                hidden = max(0, hidden - 1)
            elif re.match(r"<\s*br\s*/?\s*>", token.content, re.IGNORECASE) and not hidden:
                parts.append("\n")
        elif not hidden and token.type in {"text", "code_inline"}:
            parts.append(token.content)
        elif not hidden and token.type in {"softbreak", "hardbreak"}:
            parts.append("\n")
    return "".join(parts).strip()


def markdown_to_text(markdown):
    tokens = MarkdownIt("commonmark", {"html": True}).parse(markdown)
    paragraphs = []
    heading = False
    for token in tokens:
        if token.type == "heading_open":
            heading = True
        elif token.type == "heading_close":
            heading = False
        elif token.type == "inline":
            text = inline_text(token.children)
            if heading and text.casefold().rstrip(":") in APPENDIX_HEADINGS:
                break
            if text:
                paragraphs.append(text)
        elif token.type == "html_block":
            parser = HTMLText()
            parser.feed(token.content)
            text = "".join(parser.parts).strip()
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs) + ("\n" if paragraphs else "")


def main():
    parser = argparse.ArgumentParser(description="Convert Markdown main text to a TTS input file.")
    parser.add_argument("input", type=Path, help="Markdown input file")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: input name + -TTS.txt)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()

    input_path = args.input.expanduser()
    output_path = args.output.expanduser() if args.output else input_path.with_name(input_path.stem + "-TTS.txt")
    if input_path.resolve() == output_path.resolve():
        parser.error("input and output must be different files")
    try:
        text = markdown_to_text(input_path.read_text(encoding="utf-8"))
        with output_path.open("w" if args.force else "x", encoding="utf-8") as output:
            output.write(text)
    except (OSError, UnicodeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()