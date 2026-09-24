"""
EDA & XGBoost Performance Analysis — AIML Dataset.csv
PaySim Financial Fraud Detection Dataset
Author: Auto-generated
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# ─── Config ──────────────────────────────────────────────────────────────────
DATASET_PATH = "/Users/ghanishtrajoria/Desktop/new project/AIML Dataset.csv"
OUTPUT_DIR   = "/Users/ghanishtrajoria/Desktop/new project/eda_xgboost_results"
SAMPLE_SIZE  = 200_000   # subsample for speed; use None for full dataset
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
STYLE = "dark_background"
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor": "#1a1a2e",
    "figure.facecolor": "#0f0f1a",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#3a3a5c",
    "grid.color": "#2a2a4a",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})
PALETTE = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff", "#ff9f43"]

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  AIML Dataset — EDA & XGBoost Analysis")
print("=" * 65)
print(f"\n[1] Loading dataset from:\n    {DATASET_PATH}")
df_full = pd.read_csv(DATASET_PATH)
print(f"    Full shape : {df_full.shape[0]:,} rows × {df_full.shape[1]} columns")

if SAMPLE_SIZE and len(df_full) > SAMPLE_SIZE:
    df = df_full.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"    Sampled    : {len(df):,} rows (stratified not needed — using random)")
else:
    df = df_full.copy()
    print("    Using full dataset")

# ─── 2. Basic Info ────────────────────────────────────────────────────────────
print("\n[2] Basic Info")
print(df.dtypes.to_string())
print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
print(f"\nDuplicates: {df.duplicated().sum():,}")
print(f"\nDescribe (numeric):\n{df.describe().to_string()}")

# ─── 3. Class Distribution ───────────────────────────────────────────────────
print("\n[3] Class Distribution")
vc = df['isFraud'].value_counts()
print(f"    Non-Fraud : {vc[0]:,}  ({vc[0]/len(df)*100:.2f}%)")
print(f"    Fraud     : {vc[1]:,}  ({vc[1]/len(df)*100:.2f}%)")

# ─── 4. EDA Plots ─────────────────────────────────────────────────────────────
print("\n[4] Generating EDA plots …")

# ── Plot 1: Overview dashboard ──
fig = plt.figure(figsize=(20, 14))
fig.suptitle("AIML Dataset — Exploratory Data Analysis", fontsize=22, fontweight='bold',
             color='#00d4ff', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 4a Class balance
ax1 = fig.add_subplot(gs[0, 0])
labels = ['Non-Fraud', 'Fraud']
sizes  = [vc[0], vc[1]]
colors = ['#00d4ff', '#ff6b6b']
wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.2f%%',
                                    colors=colors, startangle=90,
                                    wedgeprops=dict(edgecolor='#0f0f1a', linewidth=2))
for at in autotexts: at.set_fontsize(11)
ax1.set_title("Class Distribution", fontsize=13, fontweight='bold')

# 4b Transaction type distribution
ax2 = fig.add_subplot(gs[0, 1])
type_counts = df['type'].value_counts()
bars = ax2.bar(type_counts.index, type_counts.values,
                color=PALETTE[:len(type_counts)], edgecolor='#0f0f1a', linewidth=1.2)
ax2.set_title("Transaction Type", fontsize=13, fontweight='bold')
ax2.set_xlabel("Type"); ax2.set_ylabel("Count")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)

# 4c Fraud by type
ax3 = fig.add_subplot(gs[0, 2])
fraud_by_type = df.groupby('type')['isFraud'].mean() * 100
colors_by_type = [PALETTE[i % len(PALETTE)] for i in range(len(fraud_by_type))]
bars3 = ax3.bar(fraud_by_type.index, fraud_by_type.values,
                 color=colors_by_type, edgecolor='#0f0f1a', linewidth=1.2)
ax3.set_title("Fraud Rate by Type (%)", fontsize=13, fontweight='bold')
ax3.set_xlabel("Type"); ax3.set_ylabel("Fraud %")
for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

# 4d Amount distribution (log scale)
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.hist(df[df['isFraud']==0]['amount'], bins=80, alpha=0.7,
         color='#00d4ff', label='Non-Fraud', edgecolor='none')
ax4.hist(df[df['isFraud']==1]['amount'], bins=80, alpha=0.85,
         color='#ff6b6b', label='Fraud', edgecolor='none')
ax4.set_yscale('log')
ax4.set_title("Transaction Amount Distribution (log scale)", fontsize=13, fontweight='bold')
ax4.set_xlabel("Amount"); ax4.set_ylabel("Count (log)")
ax4.legend()

# 4e Step (time) vs fraud density
ax5 = fig.add_subplot(gs[1, 2])
fraud_by_step = df.groupby('step')['isFraud'].mean() * 100
ax5.plot(fraud_by_step.index, fraud_by_step.values, color='#ff6b6b', linewidth=1.5)
ax5.fill_between(fraud_by_step.index, fraud_by_step.values, alpha=0.3, color='#ff6b6b')
ax5.set_title("Fraud Rate Over Time (step)", fontsize=13, fontweight='bold')
ax5.set_xlabel("Step"); ax5.set_ylabel("Fraud %")

# 4f Balance diff features
ax6 = fig.add_subplot(gs[2, 0:2])
df['orig_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['dest_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
ax6.scatter(df[df['isFraud']==0]['orig_diff'], df[df['isFraud']==0]['dest_diff'],
            alpha=0.05, s=1, c='#00d4ff', label='Non-Fraud')
ax6.scatter(df[df['isFraud']==1]['orig_diff'], df[df['isFraud']==1]['dest_diff'],
            alpha=0.4, s=3, c='#ff6b6b', label='Fraud')
ax6.set_xlim(-1e6, 3e6); ax6.set_ylim(-1e6, 3e6)
ax6.set_title("Balance Diff: Orig vs Dest", fontsize=13, fontweight='bold')
ax6.set_xlabel("Orig Balance Diff"); ax6.set_ylabel("Dest Balance Diff")
ax6.legend(markerscale=5)

# 4g Boxplot amount by fraud
ax7 = fig.add_subplot(gs[2, 2])
data_box = [df[df['isFraud']==0]['amount'].values, df[df['isFraud']==1]['amount'].values]
bp = ax7.boxplot(data_box, patch_artist=True, notch=False,
                  medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], ['#00d4ff', '#ff6b6b']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax7.set_yscale('log')
ax7.set_xticklabels(['Non-Fraud', 'Fraud'])
ax7.set_title("Amount Distribution (log)", fontsize=13, fontweight='bold')

plt.savefig(f"{OUTPUT_DIR}/01_eda_overview.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 01_eda_overview.png")

# ── Plot 2: Correlation heatmap ──
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#1a1a2e')
num_cols = ['step','amount','oldbalanceOrg','newbalanceOrig',
            'oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud',
            'orig_diff','dest_diff']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt='.2f',
            linewidths=0.5, ax=ax, annot_kws={"size": 9},
            cbar_kws={'shrink': 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight='bold',
              color='#00d4ff', pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 02_correlation_heatmap.png")

# ─── 5. Feature Engineering ───────────────────────────────────────────────────
print("\n[5] Feature Engineering")

def build_features(data):
    d = data.copy()
    d['type_enc'] = d['type'].map({'PAYMENT':0, 'TRANSFER':1, 'CASH_OUT':2,
                                    'DEBIT':3, 'CASH_IN':4}).fillna(-1).astype(int)
    d['orig_diff']         = d['oldbalanceOrg']   - d['newbalanceOrig']
    d['dest_diff']         = d['newbalanceDest']   - d['oldbalanceDest']
    d['orig_balance_ratio']= np.where(d['oldbalanceOrg'] > 0,
                                       d['newbalanceOrig'] / (d['oldbalanceOrg'] + 1), 0)
    d['dest_balance_ratio']= np.where(d['oldbalanceDest'] > 0,
                                       d['newbalanceDest'] / (d['oldbalanceDest'] + 1), 0)
    d['amount_orig_ratio'] = np.where(d['oldbalanceOrg'] > 0,
                                       d['amount'] / (d['oldbalanceOrg'] + 1), 0)
    d['surp_orig']  = (d['orig_diff'] - d['amount']).abs()
    d['surp_dest']  = (d['dest_diff'] - d['amount']).abs()
    d['log_amount'] = np.log1p(d['amount'])
    return d

df = build_features(df)
FEATURES = ['step','type_enc','amount','log_amount','oldbalanceOrg','newbalanceOrig',
            'oldbalanceDest','newbalanceDest','orig_diff','dest_diff',
            'orig_balance_ratio','dest_balance_ratio','amount_orig_ratio',
            'surp_orig','surp_dest']
TARGET = 'isFraud'
print(f"    Features used : {len(FEATURES)}")
print(f"    Features       : {FEATURES}")

X = df[FEATURES].values
y = df[TARGET].values

# ─── 6. Train / Test Split ────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\n[6] Train/Test Split")
print(f"    Train : {len(X_train):,}  | Fraud: {y_train.sum():,}")
print(f"    Test  : {len(X_test):,}   | Fraud: {y_test.sum():,}")

# ─── 7. XGBoost ───────────────────────────────────────────────────────────────
print("\n[7] Training XGBoost …")
try:
    import xgboost as xgb
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"    scale_pos_weight = {scale_pos_weight:.1f}  (class imbalance handle)")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='aucpr',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)
print("    XGBoost training complete!")

# ─── 8. Evaluation ────────────────────────────────────────────────────────────
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve, f1_score)
print("\n[8] Evaluation Metrics")
y_prob  = model.predict_proba(X_test)[:, 1]
y_pred  = model.predict(X_test)

accuracy   = (y_pred == y_test).mean()
roc_auc    = roc_auc_score(y_test, y_prob)
ap_score   = average_precision_score(y_test, y_prob)
f1         = f1_score(y_test, y_pred)
cm         = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n    Accuracy          : {accuracy*100:.4f}%")
print(f"    ROC-AUC           : {roc_auc:.4f}")
print(f"    Avg Precision (PR): {ap_score:.4f}")
print(f"    F1-Score          : {f1:.4f}")
print(f"    Confusion Matrix  :\n      TN={tn:,}  FP={fp:,}\n      FN={fn:,}  TP={tp:,}")
print(f"\n    Classification Report:\n{classification_report(y_test, y_pred, target_names=['Non-Fraud','Fraud'])}")

# ─── 9. Performance Plots ─────────────────────────────────────────────────────
print("\n[9] Generating Performance Plots …")

fig = plt.figure(figsize=(22, 14))
fig.suptitle("XGBoost — Model Performance Dashboard", fontsize=22, fontweight='bold',
             color='#ffd93d', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax1.plot(fpr, tpr, color='#00d4ff', lw=2, label=f'AUC = {roc_auc:.4f}')
ax1.plot([0,1],[0,1], 'k--', lw=1)
ax1.fill_between(fpr, tpr, alpha=0.15, color='#00d4ff')
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve", fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')

# PR Curve
ax2 = fig.add_subplot(gs[0, 1])
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ax2.plot(rec, prec, color='#ff6b6b', lw=2, label=f'AP = {ap_score:.4f}')
ax2.fill_between(rec, prec, alpha=0.15, color='#ff6b6b')
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')

# Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax3.imshow(cm_pct, cmap='YlOrRd')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Pred: 0','Pred: 1'])
ax3.set_yticklabels(['Actual: 0','Actual: 1'])
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                 ha='center', va='center', fontsize=12, color='white', fontweight='bold')
ax3.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, shrink=0.8)

# Feature Importances
ax4 = fig.add_subplot(gs[1, 0:2])
importances = model.feature_importances_
fi_df = pd.DataFrame({'feature': FEATURES, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=True).tail(15)
colors_fi = [PALETTE[i % len(PALETTE)] for i in range(len(fi_df))]
bars_fi = ax4.barh(fi_df['feature'], fi_df['importance'],
                    color=colors_fi[::-1], edgecolor='none')
ax4.set_xlabel("Importance (F-score)"); 
ax4.set_title("Top Feature Importances", fontsize=14, fontweight='bold')
for bar in bars_fi:
    ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=8)

# Score distribution
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(y_prob[y_test == 0], bins=60, alpha=0.7, color='#00d4ff',
         label='Non-Fraud', density=True, range=(0,1))
ax5.hist(y_prob[y_test == 1], bins=60, alpha=0.7, color='#ff6b6b',
         label='Fraud', density=True, range=(0,1))
ax5.set_xlabel("Predicted Probability"); ax5.set_ylabel("Density")
ax5.set_title("Prediction Score Distribution", fontsize=14, fontweight='bold')
ax5.legend()

plt.savefig(f"{OUTPUT_DIR}/03_xgboost_performance.png", dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    Saved: 03_xgboost_performance.png")

# ─── 10. Summary Report ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  Dataset       : {DATASET_PATH.split('/')[-1]}")
print(f"  Total rows    : 6,362,620 (sampled: {len(df):,})")
print(f"  Fraud rate    : {vc[1]/len(df)*100:.2f}%")
print(f"  Features      : {len(FEATURES)}")
print(f"  Model         : XGBoost (n_estimators=300, max_depth=6)")
print(f"  Accuracy      : {accuracy*100:.4f}%")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  Avg Precision : {ap_score:.4f}")
print(f"  F1-Score      : {f1:.4f}")
print(f"  Outputs saved : {OUTPUT_DIR}/")
print("=" * 65)

# Save metrics to txt
with open(f"{OUTPUT_DIR}/metrics_summary.txt", 'w') as f:
    f.write("AIML Dataset — XGBoost Performance Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Dataset rows (sampled) : {len(df):,}\n")
    f.write(f"Fraud rate             : {vc[1]/len(df)*100:.4f}%\n")
    f.write(f"Features               : {len(FEATURES)}\n\n")
    f.write(f"Accuracy               : {accuracy*100:.4f}%\n")
    f.write(f"ROC-AUC                : {roc_auc:.4f}\n")
    f.write(f"Average Precision (PR) : {ap_score:.4f}\n")
    f.write(f"F1-Score               : {f1:.4f}\n\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"  TN={tn:,}  FP={fp:,}\n  FN={fn:,}  TP={tp:,}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Non-Fraud','Fraud']))
    f.write(f"\nTop Feature Importances:\n")
    for _, row in fi_df.sort_values('importance', ascending=False).iterrows():
        f.write(f"  {row['feature']:<25} {row['importance']:.4f}\n")

print(f"\n  metrics_summary.txt also saved to {OUTPUT_DIR}/")
print("\nDone! ✓")
