# Automatic Ticket Prioritization System

This project trains a text classifier to assign priority levels (Low / Medium / High / Critical, or dataset-provided labels) to support tickets based on their subject and body.

## Setup

1. Ensure Python 3.10+ is installed (3.12 recommended for Windows wheels).
2. (Optional) Create a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux bash
# source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Source

The script downloads rows from Hugging Face Datasets Server:

- Endpoint: `https://datasets-server.huggingface.co/rows?dataset=Tobi-Bueck%2Fcustomer-support-tickets&config=default&split=train`
- You can adjust how many rows are fetched by changing `--max-rows` and `--page-size` (CLI flags) or editing defaults in `ticket_prioritization.py`.

## Run (Terminal)

- Standard run (train/evaluate + bar chart):
```bash
python ticket_prioritization.py
```

- Faster run (smaller sample, no plot):
```bash
python ticket_prioritization.py --max-rows 200 --no-plot
```

- Classify and enqueue a single ticket (no UI):
```bash
python ticket_prioritization.py --max-rows 200 --no-plot --new-ticket "SERVER DOWN for region EU, payments failing immediately!!"
```

- Interactive console mode (type tickets, manage queue):
```bash
python ticket_prioritization.py --max-rows 200 --no-plot --interactive
```
  - Commands during interactive mode:
    - `/s` to show queue summary
    - `/q` to quit

Queue persistence: predictions are appended to `priority_queue.csv` and sorted by urgency (Critical > High > Medium > Low) and timestamp.

## Run (Streamlit UI)

- Start the UI:
```bash
python -m streamlit run streamlit_app.py
```
  - If using a venv on Windows and PATH issues occur:
```bash
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

What the UI does:
- Trains/caches the model on first run (up to 500 rows).
- Text area to input a ticket, predict priority, and optionally enqueue.
- Sidebar shows current queues grouped by `Critical`, `High`, `Medium`, `Low` (top items).

## Inference (from code)

Use `assign_priority(text)` inside `ticket_prioritization.py` to predict the priority for new ticket text. The script demonstrates applying it across the dataset.

## Notes

- If the dataset contains different label names (e.g., Sev1–Sev4, P1–P4), the script normalizes common variants to canonical labels when possible.
- For larger/different datasets, adjust the normalization map and TF‑IDF parameters as needed.
