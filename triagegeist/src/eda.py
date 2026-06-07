"""
Triagegeist — Clinical EDA

Generates a comprehensive EDA report with clinical lens.
Focus on what matters for judging: clinical relevance, bias patterns, feature predictiveness.

Run: uv run python src/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    ASSETS_DIR,
    DATA_DIR,
    TRAIN_FILE,
    TEST_FILE,
    CC_FILE,
    HISTORY_FILE,
    TARGET_COL,
    VITAL_COLS,
    SCORE_COLS,
    BODY_COLS,
    HX_COLS,
    ESI_LABELS,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)
ASSETS_DIR.mkdir(exist_ok=True)

ESI_COLORS = {1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#388e3c", 5: "#1976d2"}
ESI_NAMES = [f"ESI-{k}\n({v})" for k, v in ESI_LABELS.items()]


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    cc = pd.read_csv(CC_FILE)
    hist = pd.read_csv(HISTORY_FILE)
    return train, test, cc, hist


# ---------------------------------------------------------------------------
# 1. Target distribution
# ---------------------------------------------------------------------------


def plot_target_distribution(train: pd.DataFrame) -> dict[int, int]:
    vc = train[TARGET_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ESI_NAMES, vc.values, color=[ESI_COLORS[i] for i in vc.index], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vc.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 200,
            f"{v:,}\n({v / len(train) * 100:.1f}%)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Triage Acuity Distribution (ESI 1-5)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Patient Count")
    ax.set_xlabel("ESI Level")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    return vc.to_dict()


# ---------------------------------------------------------------------------
# 2. NEWS2 and GCS — the key clinical scores
# ---------------------------------------------------------------------------


def plot_clinical_scores(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # NEWS2 by acuity
    for level in sorted(train[TARGET_COL].unique()):
        subset = train[train[TARGET_COL] == level]["news2_score"]
        axes[0].hist(subset, bins=range(0, 22), alpha=0.5, label=f"ESI-{level}", color=ESI_COLORS[level], density=True)
    axes[0].set_title("NEWS2 Score by Acuity", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("NEWS2 Score")
    axes[0].legend()

    # GCS by acuity
    for level in sorted(train[TARGET_COL].unique()):
        subset = train[train[TARGET_COL] == level]["gcs_total"]
        axes[1].hist(subset, bins=range(3, 17), alpha=0.5, label=f"ESI-{level}", color=ESI_COLORS[level], density=True)
    axes[1].set_title("GCS by Acuity", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Glasgow Coma Scale")
    axes[1].legend()

    # NEWS2 boxplot
    train.boxplot(
        column="news2_score",
        by=TARGET_COL,
        ax=axes[2],
        boxprops=dict(color="steelblue"),
        medianprops=dict(color="crimson", linewidth=2),
    )
    axes[2].set_title("NEWS2 Score by Acuity (Box)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("ESI Level")
    axes[2].set_ylabel("NEWS2")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "02_clinical_scores.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Vital signs heatmap
# ---------------------------------------------------------------------------


def plot_vitals_heatmap(train: pd.DataFrame) -> None:
    all_cols = VITAL_COLS + SCORE_COLS + ["shock_index"]
    medians = train.groupby(TARGET_COL)[all_cols].median()

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        medians.T, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5, ax=ax, cbar_kws={"label": "Median Value"}
    )
    ax.set_xlabel("ESI Acuity Level")
    ax.set_title("Median Vital Signs & Scores by Triage Acuity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "03_vitals_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Missingness analysis — clinically informative
# ---------------------------------------------------------------------------


def plot_missingness(train: pd.DataFrame) -> None:
    miss_by_acuity = {}
    miss_cols = [c for c in train.columns if train[c].isnull().any()]
    miss_cols.append("pain_score_missing")  # synthetic
    train_tmp = train.copy()
    train_tmp["pain_score_missing"] = (train_tmp["pain_score"] == -1).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall missing
    miss = train.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    miss_pct = miss / len(train) * 100
    axes[0].barh(miss_pct.index, miss_pct.values, color="steelblue")
    for i, (idx, v) in enumerate(miss_pct.items()):
        axes[0].text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    axes[0].set_title("Missing Values in Train", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Missing (%)")

    # Missingness by acuity — is it informative?
    bp_miss = train["systolic_bp"].isnull()
    rr_miss = train["respiratory_rate"].isnull()
    pain_miss = train["pain_score"] == -1

    miss_rates = (
        pd.DataFrame(
            {
                "BP missing": train.groupby(TARGET_COL).apply(lambda x: x["systolic_bp"].isnull().mean()),
                "RR missing": train.groupby(TARGET_COL).apply(lambda x: x["respiratory_rate"].isnull().mean()),
                "Pain not recorded": train.groupby(TARGET_COL).apply(lambda x: (x["pain_score"] == -1).mean()),
            }
        )
        * 100
    )
    miss_rates.plot(kind="bar", ax=axes[1], color=["#e57373", "#81c784", "#64b5f6"])
    axes[1].set_title("Missingness Rate by Acuity Level", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Missing (%)")
    axes[1].set_xlabel("ESI Level")
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "04_missingness.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Inter-rater variability (nurse/site bias)
# ---------------------------------------------------------------------------


def plot_nurse_site_variability(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Acuity by site
    site_acuity = train.groupby("site_id")[TARGET_COL].value_counts(normalize=True).unstack(fill_value=0) * 100
    site_acuity.plot(
        kind="bar", stacked=True, ax=axes[0], color=[ESI_COLORS[i] for i in sorted(train[TARGET_COL].unique())]
    )
    axes[0].set_title("Acuity Distribution by Site", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Percentage")
    axes[0].legend(title="ESI", labels=[f"ESI-{i}" for i in range(1, 6)], fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    # Acuity mean by top 20 nurses (inter-rater variability)
    nurse_mean = train.groupby("triage_nurse_id")[TARGET_COL].agg(["mean", "std", "count"])
    nurse_mean = nurse_mean[nurse_mean["count"] >= 100].sort_values("mean")
    axes[1].barh(range(len(nurse_mean)), nurse_mean["mean"].values, color="steelblue", alpha=0.7)
    axes[1].errorbar(
        nurse_mean["mean"].values,
        range(len(nurse_mean)),
        xerr=nurse_mean["std"].values,
        fmt="none",
        color="black",
        alpha=0.3,
    )
    axes[1].set_yticks(range(len(nurse_mean)))
    axes[1].set_yticklabels(nurse_mean.index, fontsize=7)
    axes[1].set_title("Mean Acuity by Nurse (±1 SD)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean ESI Level (lower = more severe triage)")
    axes[1].axvline(train[TARGET_COL].mean(), color="red", linestyle="--", alpha=0.5, label="Overall mean")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "05_interrater_variability.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Demographic bias analysis
# ---------------------------------------------------------------------------


def plot_demographic_bias(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # By sex
    sex_acuity = train.groupby("sex")[TARGET_COL].value_counts(normalize=True).unstack(fill_value=0) * 100
    sex_acuity.plot(kind="bar", stacked=True, ax=axes[0, 0], color=[ESI_COLORS[i] for i in range(1, 6)])
    axes[0, 0].set_title("Acuity by Sex", fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].legend(title="ESI", fontsize=7)

    # By age_group
    age_acuity = train.groupby("age_group")[TARGET_COL].value_counts(normalize=True).unstack(fill_value=0) * 100
    age_order = ["pediatric", "young_adult", "middle_aged", "elderly"]
    age_acuity = age_acuity.reindex(age_order)
    age_acuity.plot(kind="bar", stacked=True, ax=axes[0, 1], color=[ESI_COLORS[i] for i in range(1, 6)])
    axes[0, 1].set_title("Acuity by Age Group", fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].legend(title="ESI", fontsize=7)

    # By language
    lang_acuity = train.groupby("language")[TARGET_COL].value_counts(normalize=True).unstack(fill_value=0) * 100
    lang_acuity.plot(kind="bar", stacked=True, ax=axes[1, 0], color=[ESI_COLORS[i] for i in range(1, 6)])
    axes[1, 0].set_title("Acuity by Language", fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend(title="ESI", fontsize=7)

    # By insurance
    ins_acuity = train.groupby("insurance_type")[TARGET_COL].value_counts(normalize=True).unstack(fill_value=0) * 100
    ins_acuity.plot(kind="bar", stacked=True, ax=axes[1, 1], color=[ESI_COLORS[i] for i in range(1, 6)])
    axes[1, 1].set_title("Acuity by Insurance Type", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].legend(title="ESI", fontsize=7)

    plt.suptitle("Demographic Bias Analysis — Triage Acuity Distribution", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "06_demographic_bias.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Chief complaint analysis
# ---------------------------------------------------------------------------


def plot_chief_complaints(train: pd.DataFrame, cc: pd.DataFrame) -> None:
    merged = train[["patient_id", TARGET_COL]].merge(cc, on="patient_id")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # System by acuity
    sys_acuity = merged.groupby("chief_complaint_system")[TARGET_COL].mean().sort_values()
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sys_acuity)))
    axes[0].barh(sys_acuity.index, sys_acuity.values, color=colors)
    axes[0].set_title("Mean Acuity by Chief Complaint System", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean ESI (lower = more severe)")
    axes[0].axvline(train[TARGET_COL].mean(), color="red", linestyle="--", alpha=0.5, label="Overall mean")
    axes[0].legend()

    # Complaint word length by acuity
    merged["cc_word_count"] = merged["chief_complaint_raw"].str.split().str.len()
    merged.boxplot(
        column="cc_word_count",
        by=TARGET_COL,
        ax=axes[1],
        boxprops=dict(color="steelblue"),
        medianprops=dict(color="crimson", linewidth=2),
    )
    axes[1].set_title("Chief Complaint Length by Acuity", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("ESI Level")
    axes[1].set_ylabel("Word Count")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "07_chief_complaints.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 8. Feature correlation with target
# ---------------------------------------------------------------------------


def compute_feature_correlations(train: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    merged = train.merge(hist, on="patient_id", how="left")
    num_cols = (
        VITAL_COLS
        + SCORE_COLS
        + BODY_COLS
        + ["age", "num_prior_ed_visits_12m", "num_prior_admissions_12m", "num_active_medications", "num_comorbidities"]
        + HX_COLS
    )
    corrs = {}
    for col in num_cols:
        if col in merged.columns:
            c = merged[col].corr(merged[TARGET_COL])
            if not np.isnan(c):
                corrs[col] = c
    corr_df = pd.DataFrame({"feature": list(corrs.keys()), "corr_with_acuity": list(corrs.values())})
    corr_df["abs_corr"] = corr_df["corr_with_acuity"].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False)
    return corr_df


def plot_feature_correlations(corr_df: pd.DataFrame) -> None:
    top = corr_df.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d32f2f" if c < 0 else "#1976d2" for c in top["corr_with_acuity"]]
    ax.barh(top["feature"][::-1], top["corr_with_acuity"][::-1], color=colors[::-1])
    ax.set_title("Top 25 Features — Correlation with ESI Acuity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation (negative = lower ESI = more severe)")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "08_feature_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 9. Train vs Test drift check
# ---------------------------------------------------------------------------


def check_drift(train: pd.DataFrame, test: pd.DataFrame) -> None:
    num_cols = [
        c
        for c in train.select_dtypes(include="number").columns
        if c not in [TARGET_COL, "ed_los_hours"] and c in test.columns
    ]
    from scipy.stats import ks_2samp

    drift_results = []
    for col in num_cols:
        tr_vals = train[col].dropna()
        te_vals = test[col].dropna()
        if len(tr_vals) > 0 and len(te_vals) > 0:
            stat, pval = ks_2samp(tr_vals, te_vals)
            drift_results.append({"feature": col, "ks_stat": stat, "p_value": pval})
    drift_df = pd.DataFrame(drift_results).sort_values("ks_stat", ascending=False)
    print("\n── Train vs Test Drift (top 10 KS stats) ──")
    for _, row in drift_df.head(10).iterrows():
        flag = "⚠️" if row["ks_stat"] > 0.05 else "✅"
        print(f"  {flag} {row['feature']}: KS={row['ks_stat']:.4f} (p={row['p_value']:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("TRIAGEGEIST — CLINICAL EDA")
    print("=" * 60)

    train, test, cc, hist = load_all()

    print("\n── 1. Target Distribution ──")
    vc = plot_target_distribution(train)
    for level, count in sorted(vc.items()):
        print(f"  ESI-{level}: {count:,} ({count / len(train) * 100:.1f}%)")

    print("\n── 2. Clinical Scores ──")
    plot_clinical_scores(train)
    for level in sorted(train[TARGET_COL].unique()):
        s = train[train[TARGET_COL] == level]
        print(
            f"  ESI-{level}: NEWS2={s['news2_score'].mean():.1f}±{s['news2_score'].std():.1f}, "
            f"GCS={s['gcs_total'].mean():.1f}±{s['gcs_total'].std():.1f}"
        )

    print("\n── 3. Vitals Heatmap ──")
    plot_vitals_heatmap(train)
    print("  Saved: assets/03_vitals_heatmap.png")

    print("\n── 4. Missingness Analysis ──")
    plot_missingness(train)
    for col in ["systolic_bp", "respiratory_rate", "temperature_c"]:
        n = train[col].isnull().sum()
        print(f"  {col}: {n:,} missing ({n / len(train) * 100:.1f}%)")
    print(
        f"  pain_score == -1: {(train['pain_score'] == -1).sum():,} ({(train['pain_score'] == -1).mean() * 100:.1f}%)"
    )

    print("\n── 5. Inter-Rater Variability ──")
    plot_nurse_site_variability(train)
    nurse_means = train.groupby("triage_nurse_id")[TARGET_COL].mean()
    print(
        f"  Nurse mean acuity range: {nurse_means.min():.2f} - {nurse_means.max():.2f} (overall: {train[TARGET_COL].mean():.2f})"
    )
    site_means = train.groupby("site_id")[TARGET_COL].mean()
    print(f"  Site mean acuity range:  {site_means.min():.2f} - {site_means.max():.2f}")

    print("\n── 6. Demographic Bias ──")
    plot_demographic_bias(train)
    for col in ["sex", "age_group", "language", "insurance_type"]:
        g = train.groupby(col)[TARGET_COL].mean()
        print(f"  {col}: {dict(g.round(2))}")

    print("\n── 7. Chief Complaints ──")
    plot_chief_complaints(train, cc)
    print(f"  Unique complaints: {cc['chief_complaint_raw'].nunique():,}")
    print(f"  Systems: {cc['chief_complaint_system'].nunique()}")

    print("\n── 8. Feature Correlations ──")
    corr_df = compute_feature_correlations(train, hist)
    plot_feature_correlations(corr_df)
    print("  Top 10 features by |correlation| with acuity:")
    for _, row in corr_df.head(10).iterrows():
        direction = "↑severe" if row["corr_with_acuity"] < 0 else "↓mild"
        print(f"    {row['feature']}: r={row['corr_with_acuity']:.3f} ({direction})")

    print("\n── 9. Train vs Test Drift ──")
    check_drift(train, test)

    print("\n" + "=" * 60)
    print("EDA complete. Plots saved to assets/")
    print("=" * 60)


if __name__ == "__main__":
    main()
