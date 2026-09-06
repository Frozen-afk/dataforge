#!/usr/bin/env python3
"""
Render docs/concept-summary.md to a print-ready HTML page, then to PDF.

The submission asks for the one-page concept summary as a PDF. The Markdown in
concept-summary.md is the single source of text; this script only sets it, so
editing the summary never means editing two files.

    python docs/build-summary-pdf.py            # writes docs/concept-summary.html
    python docs/build-summary-pdf.py --pdf      # also tries to produce the PDF

PDF generation needs one of: weasyprint, or a headless Chromium/Chrome. If none
is present the script says so and leaves the HTML, which any browser can print
to PDF with "Save as PDF" at A4, default margins.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "concept-summary.md"
HTML_OUT = HERE / "concept-summary.html"
PDF_OUT = HERE / "concept-summary.pdf"

STYLE = """
@page { size: A4; margin: 12mm 13mm; }
:root { --ink: #14202b; --soft: #5a6b79; --rule: #ccd5dd; --mark: #0b6d7d; }
* { box-sizing: border-box; }
body {
  margin: 0; color: var(--ink); background: #fff;
  font: 8.5pt/1.30 "Source Serif 4", Georgia, "Times New Roman", serif;
  -webkit-font-smoothing: antialiased;
}
.sheet { max-width: 184mm; margin: 0 auto; }
h1 {
  font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  font-size: 15pt; line-height: 1.1; letter-spacing: -0.02em;
  margin: 0 0 2pt; font-weight: 700;
}
h2 {
  font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  font-size: 9.2pt; font-weight: 600; margin: 6pt 0 2pt;
  padding-top: 3pt; border-top: 0.6pt solid var(--rule);
  break-after: avoid;
}
p { margin: 0 0 3.6pt; }
strong { font-weight: 600; }
code {
  font-family: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  font-size: 7.9pt; background: #f0f3f5; padding: 0 1.5px;
}
blockquote {
  margin: 5pt 0; padding: 3.5pt 0 3.5pt 8pt;
  border-left: 2pt solid var(--mark); font-size: 9.1pt; line-height: 1.32;
}
blockquote p { margin: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 3.5pt 0 5pt;
  font-size: 7.6pt; break-inside: avoid;
}
th, td { text-align: left; padding: 1.8pt 5pt 1.8pt 0; border-bottom: 0.5pt solid var(--rule); }
th {
  font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  font-weight: 600; font-size: 7.1pt; color: var(--soft);
  border-bottom: 0.8pt solid var(--ink);
}
td:not(:first-child), th:not(:first-child) {
  text-align: right;
  font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  font-variant-numeric: tabular-nums;
}
table.words td:not(:first-child), table.words th:not(:first-child) {
  text-align: left; font-family: inherit;
}
.lede {
  font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  font-size: 8pt; color: var(--soft); margin: 0 0 7pt;
  padding-bottom: 3.5pt; border-bottom: 1.2pt solid var(--ink);
}
.cols { column-count: 2; column-gap: 7.5mm; column-rule: 0.4pt solid var(--rule); }
.cols > h2:first-child { margin-top: 0; padding-top: 0; border-top: 0; }
blockquote, table { break-inside: avoid; }
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Space+Grotesk:wght@600;700&"
    "family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&"
    'family=IBM+Plex+Mono:wght@400&display=swap">'
)


def inline(text: str) -> str:
    """Markdown inline spans. Escaping happens first, so tags cannot leak in."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def convert(markdown: str) -> str:
    """
    A deliberately small Markdown subset: exactly what the summary uses.

    Bringing in a converter would add a dependency to produce one page. If the
    summary grows a construct this does not handle, the construct shows up
    verbatim in the output, which is a loud failure rather than a silent one.
    """
    out: list[str] = []
    lines = markdown.split("\n")
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("# "):
            out.append(f"<h1>{inline(stripped[2:])}</h1>")
            index += 1
            continue

        if stripped.startswith("## "):
            out.append(f"<h2>{inline(stripped[3:])}</h2>")
            index += 1
            continue

        if stripped.startswith("> "):
            block = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                block.append(lines[index].strip().lstrip(">").strip())
                index += 1
            out.append(f"<blockquote><p>{inline(' '.join(block))}</p></blockquote>")
            continue

        if stripped.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1

            cells = [
                [cell.strip() for cell in row.strip("|").split("|")] for row in rows
            ]
            header = cells[0]
            body = [row for row in cells[2:]]  # row 1 is the alignment rule

            # A table whose data cells are prose, not numbers, is set left.
            numeric = all(
                all(re.fullmatch(r"[\d.,%\[\]\s−-]*", cell) for cell in row[1:])
                for row in body
            )

            head = "".join(f"<th>{inline(cell)}</th>" for cell in header)
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
                for row in body
            )
            klass = "" if numeric else ' class="words"'
            out.append(
                f"<table{klass}><thead><tr>{head}</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>"
            )
            continue

        paragraph = []
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^\s*([#>|])", lines[index]
        ):
            paragraph.append(lines[index].strip())
            index += 1
        out.append(f"<p>{inline(' '.join(paragraph))}</p>")

    return "\n".join(out)


def build_html() -> str:
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}")

    body = convert(SOURCE.read_text())

    # The title and the standfirst line stay full width; everything after them
    # is set in two columns. That measure is what keeps a 950-word briefing on
    # one page and still readable — a single 178mm line of 9.6pt serif would be
    # roughly 130 characters, far past what anyone tracks comfortably.
    title_html, standfirst, remainder = "", "", body

    if "<h1>" in body:
        end = body.index("</h1>") + len("</h1>")
        title_html = body[body.index("<h1>"):end]
        remainder = body[end:]

        match = re.match(r"\s*<p>(.*?)</p>", remainder, re.S)
        if match:
            standfirst = f'<p class="lede">{match.group(1)}</p>'
            remainder = remainder[match.end():]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Recurrent depth as an inference-time compute axis — Latent Loop Lab</title>
{FONTS}
<style>{STYLE}</style>
</head>
<body>
<div class="sheet">
{title_html}
{standfirst}
<div class="cols">
{remainder}
</div>
</div>
</body>
</html>
"""


def to_pdf() -> bool:
    """Try the available renderers in order. Returns whether a PDF was made."""
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=str(HTML_OUT)).write_pdf(str(PDF_OUT))
        print(f"Wrote {PDF_OUT} with weasyprint")
        return True
    except ImportError:
        pass
    except Exception as error:
        print(f"weasyprint failed: {error}")

    for binary in ("chromium", "chromium-browser", "google-chrome",
                   "google-chrome-stable"):
        path = shutil.which(binary)
        if not path:
            continue
        result = subprocess.run(
            [path, "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", f"--print-to-pdf={PDF_OUT}",
             HTML_OUT.as_uri()],
            capture_output=True,
        )
        if result.returncode == 0 and PDF_OUT.exists():
            print(f"Wrote {PDF_OUT} with {binary}")
            return True
        print(f"{binary} failed: {result.stderr.decode()[-400:]}")

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="also render a PDF")
    args = parser.parse_args()

    HTML_OUT.write_text(build_html())
    print(f"Wrote {HTML_OUT}")

    if args.pdf and not to_pdf():
        print(
            "\nNo PDF renderer found. Either install one:\n"
            "    pip install weasyprint\n"
            "or open docs/concept-summary.html in a browser and print to PDF\n"
            "at A4 with default margins. The page is styled for exactly that."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
