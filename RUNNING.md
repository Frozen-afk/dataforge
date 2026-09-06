# Running Latent Loop Lab

```bash
./run.sh
```

Starts the lab, starts the reference API, and opens the lab in your browser.
Press `Ctrl+C` to stop both.

| | |
|---|---|
| Lab | http://127.0.0.1:5173/#/start |
| Reference API | http://127.0.0.1:8000/docs |

## Options

```bash
./run.sh --no-backend    # lab only
./run.sh --no-open       # do not launch a browser
./run.sh --port 5180     # serve the lab on a different port
./run.sh --help
```

## Requirements

- **Node 18 or newer.** Required. Dependencies install automatically on the
  first run.
- **Python with FastAPI.** Optional. The lab computes everything in the
  browser, so if no suitable Python is found the script warns and starts the
  lab anyway.

To enable the API:

```bash
cd Backend
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## If something goes wrong

| Message | Fix |
|---|---|
| `Port 5173 is already in use` | `./run.sh --port 5180` |
| `No data bundle in Frontend/public/data` | `cd Backend && python export_web.py` |
| `Skipping the backend` | Expected without the venv. The lab still works. |
| Browser did not open | Open the URL the script prints. |

Everything else, including retraining and deployment, is in
[`README.md`](README.md).
