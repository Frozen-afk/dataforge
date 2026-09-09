#!/usr/bin/env python3
"""
Render a Markdown document in docs/ to a print-ready HTML page, then to PDF.

The submission asks for two PDFs: the one-page concept summary and the blog
post. Both are written as Markdown here and set by this script, so editing
either one never means editing two files. They differ only in page furniture,
which is what PROFILES holds: the summary is two columns on a single page, the
blog is one flowing column across as many pages as it needs.

    python docs/build-summary-pdf.py                    # summary HTML
    python docs/build-summary-pdf.py --pdf              # summary HTML + PDF
    python docs/build-summary-pdf.py --doc blog --pdf   # blog HTML + PDF

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

# Each profile is one deliverable: its Markdown source, its output stem, the
# browser title, and whether the body is set in columns. Everything else in the
# stylesheet is shared, so the two PDFs read as one submission.
PROFILES = {
    "summary": {
        "stem": "concept-summary",
        "title": "Recurrent depth as an inference-time compute axis "
                 "\u2014 Latent Loop Lab",
        "columns": True,
    },
    "blog": {
        "stem": "blog-post",
        "title": "Coverage tells you what a model has not seen "
                 "\u2014 Latent Loop Lab",
        "columns": False,
    },
}

STYLE = """
@page { size: A4; margin: 10mm 12mm; }
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
p { margin: 0 0 3.2pt; }
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
ul { margin: 0 0 4pt; padding-left: 11pt; }
li { margin: 0 0 2pt; }
blockquote, table { break-inside: avoid; }

/*
 * The blog runs long-form rather than as a single dense sheet, so it takes one
 * wider measure, larger type and real leading. Only the furniture changes; the
 * palette, the type families and the table rules stay shared so the two PDFs
 * read as one submission.
 */
.flow { font-size: 10pt; line-height: 1.45; }
.flow .sheet, .sheet.flow { max-width: 152mm; }
.flow h1 { font-size: 19pt; margin-bottom: 4pt; }
.flow h2 { font-size: 11pt; margin: 13pt 0 4pt; padding-top: 6pt; }
.flow p { margin: 0 0 7pt; }
.flow ul { margin: 0 0 8pt; padding-left: 14pt; }
.flow li { margin: 0 0 4pt; }
.flow code { font-size: 9pt; }
.flow table { font-size: 9pt; margin: 8pt 0 11pt; }
.flow th { font-size: 8.4pt; }
.flow th, .flow td { padding: 3.4pt 7pt 3.4pt 0; }
.flow .lede { font-size: 9.6pt; margin-bottom: 12pt; padding-bottom: 6pt; }
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

        if re.match(r"^[-*] ", stripped):
            items = []
            while index < len(lines) and re.match(r"^[-*] ", lines[index].strip()):
                item = [lines[index].strip()[2:]]
                index += 1
                # A wrapped bullet continues on an indented line that does not
                # itself start a new construct.
                while (
                    index < len(lines)
                    and lines[index].startswith(("  ", "\t"))
                    and lines[index].strip()
                    and not re.match(r"^[-*] ", lines[index].strip())
                ):
                    item.append(lines[index].strip())
                    index += 1
                items.append(" ".join(item))
            body = "".join(f"<li>{inline(item)}</li>" for item in items)
            out.append(f"<ul>{body}</ul>")
            continue

        paragraph = []
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^\s*([#>|]|[-*] )", lines[index]
        ):
            paragraph.append(lines[index].strip())
            index += 1
        out.append(f"<p>{inline(' '.join(paragraph))}</p>")

    return "\n".join(out)


def build_html(profile: dict, source: Path) -> str:
    if not source.exists():
        raise SystemExit(f"Missing {source}")

    body = convert(source.read_text())

    # The title and the standfirst line stay full width in both profiles.
    # For the summary everything after them is set in two columns, which is
    # what keeps a 950-word briefing on one page and still readable: a single
    # 178mm line of 9.6pt serif would run about 130 characters, far past what
    # anyone tracks comfortably. The blog flows instead, on a narrower measure.
    title_html, standfirst, remainder = "", "", body

    if "<h1>" in body:
        end = body.index("</h1>") + len("</h1>")
        title_html = body[body.index("<h1>"):end]
        remainder = body[end:]

        match = re.match(r"\s*<p>(.*?)</p>", remainder, re.S)
        if match:
            standfirst = f'<p class="lede">{match.group(1)}</p>'
            remainder = remainder[match.end():]

    if profile["columns"]:
        sheet_class, body_open, body_close = "sheet", '<div class="cols">', "</div>"
    else:
        sheet_class, body_open, body_close = "sheet flow", "", ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(profile["title"])}</title>
{FONTS}
<style>{STYLE}</style>
</head>
<body>
<div class="{sheet_class}">
{title_html}
{standfirst}
{body_open}
{remainder}
{body_close}
</div>
</body>
</html>
"""


def to_pdf(html_out: Path, pdf_out: Path) -> bool:
    """Try the available renderers in order. Returns whether a PDF was made."""
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=str(html_out)).write_pdf(str(pdf_out))
        print(f"Wrote {pdf_out} with weasyprint")
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
             "--no-pdf-header-footer", f"--print-to-pdf={pdf_out}",
             html_out.as_uri()],
            capture_output=True,
        )
        if result.returncode == 0 and pdf_out.exists():
            print(f"Wrote {pdf_out} with {binary}")
            return True
        print(f"{binary} failed: {result.stderr.decode()[-400:]}")

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="also render a PDF")
    parser.add_argument(
        "--doc",
        choices=sorted(PROFILES) + ["all"],
        default="summary",
        help="which deliverable to build (default: summary)",
    )
    args = parser.parse_args()

    names = sorted(PROFILES) if args.doc == "all" else [args.doc]
    failed = False

    for name in names:
        profile = PROFILES[name]
        source = HERE / f"{profile['stem']}.md"
        html_out = HERE / f"{profile['stem']}.html"
        pdf_out = HERE / f"{profile['stem']}.pdf"

        html_out.write_text(build_html(profile, source))
        print(f"Wrote {html_out}")

        if args.pdf and not to_pdf(html_out, pdf_out):
            print(
                f"\nNo PDF renderer found. Either install one:\n"
                f"    pip install weasyprint\n"
                f"or open {html_out.relative_to(HERE.parent)} in a browser and\n"
                f"print to PDF at A4 with default margins. The page is styled\n"
                f"for exactly that."
            )
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
