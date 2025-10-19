import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from ticket_prioritization import (
    load_dataset,
    build_text,
    normalize_priority_labels,
    PriorityModel,
    append_to_priority_queue,
    update_ticket_status,
    list_tickets_by_status,
    VALID_STATUSES,
    PRIORITY_QUEUE_PATH,
)


st.set_page_config(page_title="Ticket Prioritization", layout="wide")


def load_or_fix_queue(queue_path: str = None) -> pd.DataFrame:
    """Load the priority queue CSV and ensure expected columns exist.

    If the file is missing expected columns (like 'status', 'assigned_to',
    'notes', 'updated_at', 'timestamp' etc.), this function will add them
    with sensible defaults and overwrite the file so future reads succeed.
    """
    if queue_path is None:
        queue_path = PRIORITY_QUEUE_PATH
    
    # Handle missing file case
    if not os.path.exists(queue_path):
        qdf = pd.DataFrame(columns=[
            "text", "predicted_priority", "urgency_rank", "timestamp",
            "status", "assigned_to", "notes", "updated_at"
        ])
        qdf.to_csv(queue_path, index=False)
        return qdf
    
    qdf = pd.read_csv(queue_path)
    expected_cols = [
        "text",
        "predicted_priority",
        "urgency_rank",
        "timestamp",
        "status",
        "assigned_to",
        "notes",
        "updated_at",
    ]
    changed = False
    for col in expected_cols:
        if col not in qdf.columns:
            # sensible defaults
            if col == "status":
                qdf[col] = "new"
            elif col in {"assigned_to", "notes"}:
                qdf[col] = ""
            elif col == "updated_at":
                qdf[col] = qdf.get("timestamp", pd.Series([datetime.utcnow().isoformat()] * len(qdf)))
            elif col == "urgency_rank":
                # attempt to derive from predicted_priority if available
                if "predicted_priority" in qdf.columns:
                    qdf[col] = qdf["predicted_priority"].str.lower().map({"low": 0, "medium": 1, "high": 2, "critical": 3}).fillna(1).astype(int)
                else:
                    qdf[col] = 1
            else:
                qdf[col] = ""
            changed = True
    if changed:
        # Keep a backup then write fixed file
        try:
            backup = queue_path + ".bak"
            qdf.to_csv(backup, index=False)
        except Exception:
            pass
        qdf.to_csv(queue_path, index=False)
    return qdf

@st.cache_data(show_spinner=False)
def load_training_data(max_rows: int = 500) -> pd.DataFrame:
    return load_dataset(max_rows=max_rows, page_size=100)

@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame) -> Tuple[PriorityModel, Dict[int, str]]:
    df = df.copy()
    df["text"] = build_text(df)
    y, label_to_id, id_to_label = normalize_priority_labels(df)
    # Fallback: if fewer than 2 classes, raise
    if len(set(y.dropna().tolist())) < 2:
        raise RuntimeError("Dataset has fewer than 2 priority classes after normalization.")
    model = PriorityModel()
    model.label_to_id = label_to_id
    model.id_to_label = id_to_label
    # Simple fit on all rows for demo; for real use, keep a validation split
    model.fit(df["text"], y)
    return model, id_to_label


def render_sidebar_queues() -> None:
    st.sidebar.header("Queues")
    qdf = load_or_fix_queue()
    if not qdf.empty:
        # Group by status first, then priority
        for status in VALID_STATUSES:
            status_subset = qdf[qdf["status"].str.lower() == status]
            if status_subset.empty:
                continue
            with st.sidebar.expander(f"{status.upper()} ({len(status_subset)})", expanded=(status == "new")):
                order = ["critical", "high", "medium", "low"]
                for pr in order:
                    subset = status_subset[status_subset["predicted_priority"].str.lower() == pr]
                    if subset.empty:
                        continue
                    st.write(f"**{pr.title()}** ({len(subset)})")
                    for idx, row in subset.head(5).iterrows():
                        assigned = f" → {row['assigned_to']}" if row['assigned_to'] else ""
                        st.write(f"[{idx}] {row['text'][:60]}{'...' if len(row['text'])>60 else ''}{assigned}")
    else:
        st.sidebar.caption("Queue file not found yet.")


def main() -> None:
    st.title("Automatic Ticket Prioritization")
    with st.spinner("Loading training data and training model..."):
        df = load_training_data(max_rows=500)
        model, id_to_label = train_model(df)

    render_sidebar_queues()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Classify New Ticket")
        text = st.text_area("Enter subject + body", height=160, placeholder="e.g., Payment gateway failing since 2am; customers see 500 on checkout...")
        col1a, col1b = st.columns([1, 1])
        with col1a:
            add_to_queue = st.checkbox("Add to queue", value=True)
        with col1b:
            submit = st.button("Predict Priority")

        if submit:
            if not text.strip():
                st.warning("Please enter ticket text.")
                st.stop()
            label = model.assign_priority(text)
            st.success(f"Predicted priority: {label}")
            if add_to_queue:
                queue_df = append_to_priority_queue(text, label)
                st.info(f"Added to queue. Total tickets: {len(queue_df)}")
                st.rerun()

    with col2:
        st.subheader("Update Ticket Status")
        qdf = load_or_fix_queue()
        if not qdf.empty:
            # Show tickets that can be updated
            active_tickets = qdf[qdf["status"].isin(["new", "in_progress", "resolved"])]
            if not active_tickets.empty:
                ticket_options = []
                for idx, row in active_tickets.iterrows():
                    ticket_options.append(f"[{idx}] {row['status'].upper()} | {row['predicted_priority'].upper()} | {row['text'][:40]}...")
                
                selected_ticket = st.selectbox("Select ticket to update:", ticket_options)
                if selected_ticket:
                    ticket_id = int(selected_ticket.split("]")[0][1:])
                    
                    col2a, col2b = st.columns(2)
                    with col2a:
                        new_status = st.selectbox("New status:", VALID_STATUSES)
                    with col2b:
                        assigned_to = st.text_input("Assign to:", placeholder="e.g., john@company.com")
                    
                    notes = st.text_area("Notes:", placeholder="Optional notes...")
                    
                    if st.button("Update Status"):
                        success = update_ticket_status(ticket_id, new_status, assigned_to, notes)
                        if success:
                            st.success("Ticket updated!")
                            st.rerun()
                        else:
                            st.error("Failed to update ticket")
            else:
                st.info("No active tickets to update")
        else:
            st.info("No tickets in queue")

    st.divider()
    st.subheader("Current Queue Snapshot")
    qdf = load_or_fix_queue()
    if not qdf.empty:
        # Show only essential columns for better display
        display_cols = ["text", "predicted_priority", "status", "assigned_to", "timestamp"]
        st.dataframe(qdf[display_cols].head(50), use_container_width=True)
    else:
        st.caption("Queue empty.")


if __name__ == "__main__":
    main()
