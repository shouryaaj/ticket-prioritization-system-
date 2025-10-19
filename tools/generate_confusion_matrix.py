import os
import sys
# Ensure project root is on sys.path so imports work when running from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ticket_prioritization import load_dataset, build_text, normalize_priority_labels, PriorityModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'confusion_matrix.png')


def main():
    print('Loading dataset...')
    try:
        df = load_dataset(max_rows=2000, page_size=200)
    except Exception as e:
        print('Failed to fetch remote dataset, falling back to local CSV:', e)
        local_path = os.path.join(ROOT, 'dataset', 'Help Desk Tickets', 'issues.csv')
        if not os.path.exists(local_path):
            print('Local dataset not found at', local_path)
            return
        raw = pd.read_csv(local_path)
        # Construct minimal DataFrame matching expected columns: subject, body, priority
        df = pd.DataFrame()
        # Use issue_type and issue_proj as a short subject; use issue_created as body fallback
        if 'issue_type' in raw.columns:
            df['subject'] = raw['issue_type'].astype(str)
        else:
            df['subject'] = ''
        if 'issue_proj' in raw.columns:
            df['body'] = raw['issue_proj'].astype(str)
        else:
            df['body'] = ''
        # Column containing priority labels in the local CSV
        pr_col = None
        for candidate in ['issue_priority', 'issue_priority_code', 'priority', 'label']:
            if candidate in raw.columns:
                pr_col = candidate
                break
        df['priority'] = raw[pr_col].astype(str) if pr_col is not None else None
    if df.empty:
        print('No data loaded. Exiting.')
        return
    df['text'] = build_text(df)
    y, label_to_id, id_to_label = normalize_priority_labels(df)
    # Filter to rows with labels
    mask = y.notna()
    X = df.loc[mask, 'text']
    y = y.loc[mask]
    if len(set(y.tolist())) < 2:
        print('Need at least 2 classes to compute confusion matrix.')
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = PriorityModel()
    model.label_to_id = label_to_id
    model.id_to_label = id_to_label
    print('Training model...')
    model.fit(X_train, y_train)
    print('Predicting...')
    y_pred = model.predict(list(X_test))
    labels_sorted = [label for idx, label in sorted([(i, id_to_label[i]) for i in id_to_label.keys()])]
    cm = confusion_matrix(y_test, y_pred, labels=[label_to_id[l] for l in labels_sorted])
    print('Confusion matrix (rows=true, cols=pred):')
    print(labels_sorted)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    out = os.path.abspath(OUT_PATH)
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved confusion matrix to: {out}')


if __name__ == '__main__':
    main()
