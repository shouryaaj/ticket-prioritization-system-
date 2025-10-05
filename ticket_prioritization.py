import json
import math
import string
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import os
from datetime import datetime


DATASET_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=Tobi-Bueck%2Fcustomer-support-tickets&config=default&split=train"
)
PRIORITY_QUEUE_PATH = "priority_queue.csv"


def fetch_rows(offset: int, length: int) -> List[dict]:
    response = requests.get(DATASET_URL, params={"offset": offset, "length": length}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("rows", [])
    return rows


def load_dataset(max_rows: int = 1000, page_size: int = 100) -> pd.DataFrame:
    """Load rows from HF datasets-server endpoint into a DataFrame.

    The server returns rows with a "row" field containing columns. We try to
    extract subject/body/priority or sensible fallbacks.
    """
    collected: List[Dict[str, str]] = []
    offset = 0
    while offset < max_rows:
        to_fetch = min(page_size, max_rows - offset)
        batch = fetch_rows(offset=offset, length=to_fetch)
        if not batch:
            break
        for item in batch:
            row = item.get("row", {})
            # Try common field names
            subject = (
                row.get("subject")
                or row.get("title")
                or row.get("summary")
                or ""
            )
            body = (
                row.get("body")
                or row.get("description")
                or row.get("content")
                or row.get("text")
                or ""
            )
            priority = (
                row.get("priority")
                or row.get("severity")
                or row.get("label")
                or row.get("target")
                or row.get("category")
            )
            # If there's a single text field and no subject, treat it as body.
            if not subject and body and len(body) < 40:
                # Some datasets may have short title in body and long text elsewhere
                subject = body
                body = row.get("text") or row.get("description") or ""
            collected.append({
                "subject": str(subject) if subject is not None else "",
                "body": str(body) if body is not None else "",
                "priority": str(priority) if priority is not None else None,
            })
        offset += to_fetch
        if len(batch) < to_fetch:
            break
    df = pd.DataFrame(collected)
    # Drop rows without priority label
    df = df.dropna(subset=["priority"]).reset_index(drop=True)
    return df


def build_text(df: pd.DataFrame) -> pd.Series:
    subject = df["subject"].fillna("")
    body = df["body"].fillna("")
    combined = (subject + " " + body).str.strip()
    # Fallback to body if combined empty
    combined = combined.mask(combined.eq(""), body)
    return combined


def normalize_priority_labels(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int], Dict[int, str]]:
    raw = df["priority"].astype(str).str.strip().str.lower()
    # Map common variants to canonical
    canonical_map: Dict[str, str] = {
        "l": "low", "lo": "low", "low": "low",
        "m": "medium", "med": "medium", "medium": "medium",
        "h": "high", "hi": "high", "high": "high",
        "c": "critical", "crit": "critical", "critical": "critical",
        "urgent": "high", "immediate": "critical", "p1": "critical", "p2": "high", "p3": "medium", "p4": "low",
        "sev1": "critical", "sev2": "high", "sev3": "medium", "sev4": "low",
    }
    canonical = raw.map(lambda x: canonical_map.get(x, x))
    classes = sorted(canonical.unique())
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    y = canonical.map(label_to_id)
    return y, label_to_id, id_to_label


def get_urgency_rank(label: str) -> int:
    # Higher is more urgent
    order = ["low", "medium", "high", "critical"]
    try:
        return order.index(label)
    except ValueError:
        return 1  # default near-medium


def append_to_priority_queue(text: str, predicted_label: str, queue_path: str = PRIORITY_QUEUE_PATH) -> pd.DataFrame:
    timestamp = datetime.utcnow().isoformat()
    new_row = {
        "text": text,
        "predicted_priority": predicted_label,
        "urgency_rank": get_urgency_rank(predicted_label),
        "timestamp": timestamp,
    }
    if os.path.exists(queue_path):
        queue_df = pd.read_csv(queue_path)
    else:
        queue_df = pd.DataFrame(columns=["text", "predicted_priority", "urgency_rank", "timestamp"])
    queue_df = pd.concat([queue_df, pd.DataFrame([new_row])], ignore_index=True)
    # Sort: highest urgency_rank first, then oldest timestamp first
    queue_df = queue_df.sort_values(by=["urgency_rank", "timestamp"], ascending=[False, True]).reset_index(drop=True)
    queue_df.to_csv(queue_path, index=False)
    return queue_df


def print_queue_summary(queue_path: str = PRIORITY_QUEUE_PATH) -> None:
    if not os.path.exists(queue_path):
        print("Queue is empty.")
        return
    qdf = pd.read_csv(queue_path)
    order = ["critical", "high", "medium", "low"]
    for pr in order:
        subset = qdf[qdf["predicted_priority"].str.lower() == pr]
        print(f"\n{pr.title()} ({len(subset)})")
        for _, row in subset.head(10).iterrows():
            text = row["text"]
            print(f" - {text[:120]}{'...' if len(text)>120 else ''}")


class PriorityModel:
    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}

    def fit(self, texts: pd.Series, labels: pd.Series) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                lowercase=True,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                n_jobs=None,
            )),
        ])
        self.pipeline.fit(texts, labels)

    def predict(self, texts: List[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet")
        return self.pipeline.predict(texts)

    def assign_priority(self, text: str) -> str:
        label_id = int(self.predict([text])[0])
        return self.id_to_label.get(label_id, str(label_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic Ticket Prioritization")
    parser.add_argument("--max-rows", type=int, default=1000, help="Max rows to fetch")
    parser.add_argument("--page-size", type=int, default=100, help="Rows per request")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--new-ticket", type=str, default=None, help="New ticket text to classify and enqueue")
    parser.add_argument("--interactive", action="store_true", help="Interactive console mode to classify and queue tickets")
    args = parser.parse_args()

    print("Loading dataset from Hugging Face datasets-server ...")
    df = load_dataset(max_rows=args.max_rows, page_size=args.page_size)
    if df.empty:
        raise RuntimeError("No data loaded. Try increasing max_rows or check connectivity.")

    df["text"] = build_text(df)
    y, label_to_id, id_to_label = normalize_priority_labels(df)

    # Ensure at least two classes exist
    if len(set(y.dropna().tolist())) < 2:
        raise RuntimeError("Dataset has fewer than 2 priority classes after normalization.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, stratify=y, random_state=42
    )

    model = PriorityModel()
    model.label_to_id = label_to_id
    model.id_to_label = id_to_label

    print("Training TF-IDF + Logistic Regression model ...")
    model.fit(X_train, y_train)

    print("Evaluating on test set ...")
    y_pred = model.predict(list(X_test))
    print(classification_report(y_test, y_pred, target_names=[id_to_label[i] for i in sorted(id_to_label.keys())]))

    # Inference function
    def assign_priority(text: str) -> str:
        return model.assign_priority(text)

    # Apply to dataset
    print("Running inference across the dataset and grouping results ...")
    df["predicted_priority_id"] = model.predict(df["text"].tolist())
    df["predicted_priority"] = df["predicted_priority_id"].map(id_to_label)

    # Show grouping
    groups = df.groupby("predicted_priority")["text"].apply(list)
    for pr, texts in groups.items():
        print(f"Priority {pr} tickets:")
        for t in texts[:5]:
            print(" -", (t[:200] + ("..." if len(t) > 200 else "")))

    # Bar chart
    counts = df["predicted_priority"].value_counts()
    if not args.no_plot:
        counts.plot(kind="bar")
        plt.xlabel("Predicted Priority")
        plt.ylabel("Number of Tickets")
        plt.title("Distribution of Predicted Priority Levels")
        plt.tight_layout()
        plt.show()

    # If a new ticket is provided, classify and enqueue
    if args.new_ticket:
        predicted = assign_priority(args.new_ticket)
        queue_df = append_to_priority_queue(args.new_ticket, predicted)
        print("\nNew ticket classified and queued:")
        print(f" - Predicted priority: {predicted}")
        print(f" - Queue saved to: {PRIORITY_QUEUE_PATH}")
        print("Top of the queue (next to handle):")
        for i, row in queue_df.head(10).iterrows():
            print(f"[{row['predicted_priority']}] {row['text'][:120]}{'...' if len(row['text'])>120 else ''}")

    # Interactive mode
    if args.interactive:
        print("\nEntering interactive mode. Type a ticket and press Enter.")
        print("Commands: /q to quit, /s to show queue summary")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not line:
                continue
            if line in {"/q", "/quit", ":q"}:
                print("Bye.")
                break
            if line in {"/s", "/show", ":s"}:
                print_queue_summary()
                continue
            predicted = assign_priority(line)
            append_to_priority_queue(line, predicted)
            print(f"[{predicted}] added to queue. (/s to show)")


if __name__ == "__main__":
    main()
