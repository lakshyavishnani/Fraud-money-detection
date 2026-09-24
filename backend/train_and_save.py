"""
Train XGBoost on AIML Dataset.csv and save model + metadata artifacts.
Run once before starting the API server.
"""

import os, json, warnings
import pandas as pd
import numpy as np
import joblib
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "AIML Dataset.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
META_PATH    = os.path.join(BASE_DIR, "metrics.json")
FEAT_PATH    = os.path.join(BASE_DIR, "feature_names.json")

SAMPLE_SIZE  = 200_000
RANDOM_STATE = 42

print("=" * 60)
print("  FraudShield AI — Model Training & Save")
print("=" * 60)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print(f"\n[1] Loading dataset …")
df_full = pd.read_csv(DATASET_PATH)
print(f"    Full shape: {df_full.shape[0]:,} × {df_full.shape[1]}")
if SAMPLE_SIZE and len(df_full) > SAMPLE_SIZE:
    df = df_full.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"    Sampled: {len(df):,} rows")
else:
    df = df_full.copy()

# ── 2. Feature Engineering ─────────────────────────────────────────────────────
def build_features(data):
    d = data.copy()
    d['type_enc'] = d['type'].map({'PAYMENT':0,'TRANSFER':1,'CASH_OUT':2,
                                    'DEBIT':3,'CASH_IN':4}).fillna(-1).astype(int)
    d['orig_diff']          = d['oldbalanceOrg']   - d['newbalanceOrig']
    d['dest_diff']          = d['newbalanceDest']   - d['oldbalanceDest']
    d['orig_balance_ratio'] = np.where(d['oldbalanceOrg']  > 0,
                                        d['newbalanceOrig']  / (d['oldbalanceOrg']  + 1), 0)
    d['dest_balance_ratio'] = np.where(d['oldbalanceDest'] > 0,
                                        d['newbalanceDest'] / (d['oldbalanceDest'] + 1), 0)
    d['amount_orig_ratio']  = np.where(d['oldbalanceOrg']  > 0,
                                        d['amount'] / (d['oldbalanceOrg'] + 1), 0)
    d['surp_orig']   = (d['orig_diff'] - d['amount']).abs()
    d['surp_dest']   = (d['dest_diff'] - d['amount']).abs()
    d['log_amount']  = np.log1p(d['amount'])
    return d

FEATURES = ['step','type_enc','amount','log_amount','oldbalanceOrg','newbalanceOrig',
            'oldbalanceDest','newbalanceDest','orig_diff','dest_diff',
            'orig_balance_ratio','dest_balance_ratio','amount_orig_ratio',
            'surp_orig','surp_dest']
TARGET   = 'isFraud'

print(f"\n[2] Feature engineering …  ({len(FEATURES)} features)")
df = build_features(df)
X  = df[FEATURES].values
y  = df[TARGET].values
vc = df[TARGET].value_counts()

# ── 3. Split ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"[3] Split: train={len(X_train):,}  test={len(X_test):,}")

# ── 4. Train XGBoost ───────────────────────────────────────────────────────────
import xgboost as xgb
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"[4] Training XGBoost (scale_pos_weight={scale_pos_weight:.1f}) …")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr', random_state=RANDOM_STATE,
    n_jobs=-1, verbosity=0
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ── 5. Evaluate ────────────────────────────────────────────────────────────────
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              confusion_matrix, classification_report)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
acc    = (y_pred == y_test).mean()
roc    = roc_auc_score(y_test, y_prob)
ap     = average_precision_score(y_test, y_prob)
f1     = f1_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n[5] Metrics:")
print(f"    Accuracy   : {acc*100:.4f}%")
print(f"    ROC-AUC    : {roc:.4f}")
print(f"    Avg Prec   : {ap:.4f}")
print(f"    F1-Score   : {f1:.4f}")

# Feature importances
fi = dict(zip(FEATURES, [float(v) for v in model.feature_importances_]))
fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

# ── 6. Save artifacts ──────────────────────────────────────────────────────────
print(f"\n[6] Saving artifacts …")
joblib.dump(model, MODEL_PATH)
print(f"    Saved: {MODEL_PATH}")

with open(FEAT_PATH, 'w') as f:
    json.dump(FEATURES, f)
print(f"    Saved: {FEAT_PATH}")

metrics = {
    "accuracy":          round(acc * 100, 4),
    "roc_auc":           round(roc, 4),
    "avg_precision":     round(ap, 4),
    "f1_score":          round(f1, 4),
    "fraud_rate":        round(vc[1] / len(df) * 100, 4),
    "sample_size":       len(df),
    "confusion_matrix":  {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "feature_importances": fi_sorted
}
with open(META_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"    Saved: {META_PATH}")

print("\nDone! ✓  Model artifacts ready for the API server.")
