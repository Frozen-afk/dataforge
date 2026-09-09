# Submission package — Latent Loop Lab

DataForge 2026, Pathway Track.

**Start here:** [`SUBMISSION.md`](SUBMISSION.md) — the package index, the
one-sentence claim, and where each required item lives.

| File | What it is |
|---|---|
| [`SUBMISSION.md`](SUBMISSION.md) | Package index and judge's five-minute path |
| [`AI-DISCLOSURE.md`](AI-DISCLOSURE.md) | **Template. The team must complete it.** |
| [`SOURCES-AND-LICENSES.md`](SOURCES-AND-LICENSES.md) | Code, data, weights, fonts, libraries |
| [`docs/concept-summary.pdf`](docs/concept-summary.pdf) | The required one-page summary |
| [`docs/blog-post.pdf`](docs/blog-post.pdf) | The required blog PDF |
| [`evidence/`](evidence/) | Claim sheet, citation ledger, per-claim source matrix |
| [`artifact-build/`](artifact-build/) | Static build of the artifact, for offline review |

## Viewing the offline build

`artifact-build/` uses root-relative asset paths, so it needs a server rooted at
that directory. Opening `index.html` from the filesystem shows a blank page.

```bash
cd submission/artifact-build && python -m http.server 8080
# then open http://127.0.0.1:8080/
```

The deployed artifact is the canonical one. This snapshot exists so a judge
without network access can still open the lab, and so the reviewed build is
pinned alongside the documents that describe it.

## Regenerating this folder

Everything here is copied or derived from the repository root. The build
snapshot comes from `Frontend/dist` after `npm run build`; the PDFs come from
`docs/`; the evidence files come from `Backend/docs/`. Nothing here is a
separate source of truth, so edit the originals and re-copy.
