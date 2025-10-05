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
    PRIORITY_QUEUE_PATH,
)


st.set_page_config(page_title="Ticket Prioritization", layout="wide")

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
    if os.path.exists(PRIORITY_QUEUE_PATH):
        qdf = pd.read_csv(PRIORITY_QUEUE_PATH)
        # group by predicted_priority in desired order
        order = ["critical", "high", "medium", "low"]
        for pr in order:
            subset = qdf[qdf["predicted_priority"].str.lower() == pr]
            st.sidebar.subheader(f"{pr.title()} ({len(subset)})")
            if subset.empty:
                st.sidebar.caption("No tickets queued")
            else:
                for _, row in subset.head(10).iterrows():
                    st.sidebar.write(f"[{row['timestamp']}] {row['text'][:90]}{'...' if len(row['text'])>90 else ''}")
    else:
        st.sidebar.caption("Queue file not found yet.")


def main() -> None:
    st.title("Automatic Ticket Prioritization")
    with st.spinner("Loading training data and training model..."):
        df = load_training_data(max_rows=500)
        model, id_to_label = train_model(df)

    render_sidebar_queues()

    st.subheader("Classify New Ticket")
    text = st.text_area("Enter subject + body", height=160, placeholder="e.g., Payment gateway failing since 2am; customers see 500 on checkout...")
    col1, col2 = st.columns([1, 1])
    with col1:
        add_to_queue = st.checkbox("Add to queue", value=True)
    with col2:
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
            # Re-render sidebar to reflect latest queue
            render_sidebar_queues()

    st.divider()
    st.subheader("Current Queue Snapshot")
    if os.path.exists(PRIORITY_QUEUE_PATH):
        qdf = pd.read_csv(PRIORITY_QUEUE_PATH)
        st.dataframe(qdf.head(50))
    else:
        st.caption("Queue empty.")


if __name__ == "__main__":
    main()
