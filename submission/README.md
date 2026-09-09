# Submission package — Latent Loop Lab

DataForge 2026, Pathway Track. Team VibeCoders.

**Start here:** [`SUBMISSION.md`](SUBMISSION.md) — the package index, the
one-sentence claim, and where each required item lives.

| File | What it is |
|---|---|
| [`SUBMISSION.md`](SUBMISSION.md) | Package index and judge's five-minute path |
| [`AI-DISCLOSURE.md`](AI-DISCLOSURE.md) | AI assistance, tools, technical ownership, source verification, mentorship |
| [`SOURCES-AND-LICENSES.md`](SOURCES-AND-LICENSES.md) | Code, data, weights, fonts, libraries |
| [`docs/concept-summary.pdf`](docs/concept-summary.pdf) | The required one-page summary |
| [`docs/blog-post.pdf`](docs/blog-post.pdf) | The required blog PDF |
| [`evidence/`](evidence/) | Claim sheet, citation ledger, per-claim source matrix |

## This folder is the only copy
The two PDFs are built from the Markdown beside them:

```bash
pip install weasyprint                                     # or use headless Chrome
python submission/docs/build-summary-pdf.py --doc all --pdf
```

Edit the Markdown, rebuild, never edit a PDF directly.

## The artifact

The artifact is not a file in this folder. It is the static site built from
`Frontend/` and published by CI on every push to `main`; the URL is row 1 of
[`SUBMISSION.md`](SUBMISSION.md). To open the current source locally:

```bash
cd Frontend && npm install && npm run dev
```

No Python, no backend, no checkpoint download. Both computational layers run in
the browser from the data bundle committed under `Frontend/public/data`.
