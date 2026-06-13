"""
PCA Market Regime Analysis
Person 3: PCA implementation, comparison with autoencoder results,
visualizations, and interpretation of identified market regimes.

Inputs  (from DATA LAYER + market-regime-autoencoder/outputs/handoff/):
  - ../DATA LAYER/windows.npz          — (n_windows, 26, 30) scaled windows
  - ../DATA LAYER/metadata.json        — split date boundaries
  - ../DATA LAYER/schema.json          — feature names
  - ../market-regime-autoencoder/outputs/handoff/clustered_regimes.csv
  - ../market-regime-autoencoder/outputs/handoff/latent_space.csv

Outputs (saved to outputs/):
  - pca_latent_space.csv               — 3-component PCA projections
  - pca_clustered_regimes.csv          — PCA + KMeans regime labels
  - pca_ae_comparison.csv              — merged PCA & AE regimes for all windows
  - figures/01_scree_plot.png
  - figures/02_pca_2d_regimes.png
  - figures/03_ae_2d_regimes.png
  - figures/04_pca_vs_ae_3d.png
  - figures/05_regime_timeline.png
  - figures/06_regime_feature_profiles.png
  - figures/07_regime_agreement_heatmap.png
  - figures/08_reconstruction_loss_vs_pca_variance.png
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DATA LAYER"
AE_DIR = ROOT / "market-regime-autoencoder" / "outputs" / "handoff"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]   # regime 0-3 colours
SPLIT_COLORS = {"train": "#1a1a2e", "val": "#e94560", "test": "#0f3460"}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
})

N_COMPONENTS_3 = 3    # match AE latent dimensionality
N_CLUSTERS = 4        # match AE clustering
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_windows():
    """Return flattened windows (n, 780) and aligned date series per split."""
    windows = np.load(DATA_DIR / "windows.npz")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(DATA_DIR / "schema.json") as f:
        schema = json.load(f)

    feature_names = schema["feature_order"]

    splits = {}
    for split in ("train", "val", "test"):
        w = windows[split].astype(np.float32)           # (n, 26, 30)
        flat = w.reshape(w.shape[0], -1)                # (n, 780)
        splits[split] = flat

    # Dates come from AE CSV (it already resolved the date per window).
    ae_df = pd.read_csv(AE_DIR / "clustered_regimes.csv", parse_dates=["date"])

    dates = {}
    for split in ("train", "val", "test"):
        dates[split] = ae_df.loc[ae_df["split"] == split, "date"].values

    return splits, dates, feature_names


def load_ae_results():
    df = pd.read_csv(AE_DIR / "clustered_regimes.csv", parse_dates=["date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIT PCA
# ══════════════════════════════════════════════════════════════════════════════

def fit_pca(splits):
    """Fit full PCA on train, return (full_pca, pca_3, projections_dict)."""
    X_train = splits["train"]

    # Full PCA for explained-variance analysis
    full_pca = PCA(random_state=RANDOM_STATE)
    full_pca.fit(X_train)

    # 3-component PCA for comparison with AE
    pca_3 = PCA(n_components=N_COMPONENTS_3, random_state=RANDOM_STATE)
    pca_3.fit(X_train)

    projections = {}
    for split, X in splits.items():
        projections[split] = pca_3.transform(X)

    return full_pca, pca_3, projections


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTER PCA LATENT SPACE
# ══════════════════════════════════════════════════════════════════════════════

def cluster_pca(projections):
    """Fit KMeans on train PCA projections, assign regimes to all splits."""
    km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE)
    km.fit(projections["train"])

    labels = {}
    for split, Z in projections.items():
        labels[split] = km.predict(Z)

    return km, labels


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD OUTPUT DATAFRAMES
# ══════════════════════════════════════════════════════════════════════════════

def build_dataframes(projections, labels, dates):
    rows = []
    for split in ("train", "val", "test"):
        Z = projections[split]
        L = labels[split]
        D = dates[split]
        for i, (z, l, d) in enumerate(zip(Z, L, D)):
            rows.append({
                "split": split,
                "window_index": i,
                "date": d,
                "pc1": z[0], "pc2": z[1], "pc3": z[2],
                "regime": int(l),
            })
    return pd.DataFrame(rows)


def merge_with_ae(pca_df, ae_df):
    merged = pca_df[["split", "date", "pc1", "pc2", "pc3", "regime"]].copy()
    merged = merged.rename(columns={"regime": "pca_regime"})
    ae_cols = ae_df[["date", "z1", "z2", "z3", "regime"]].rename(
        columns={"regime": "ae_regime"}
    )
    merged = merged.merge(ae_cols, on="date", how="inner")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── 5.1  Scree plot ──────────────────────────────────────────────────────────

def fig_scree(full_pca):
    evr = full_pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA Explained Variance — Scree Plot", fontsize=14, fontweight="bold")

    # Individual
    ax = axes[0]
    ax.bar(range(1, 31), evr[:30] * 100, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Individual (first 30 PCs)")
    ax.set_xticks(range(1, 31))

    # Cumulative
    ax = axes[1]
    ax.plot(range(1, len(cumulative) + 1), cumulative * 100, "o-", color="#FF5722",
            linewidth=2, markersize=4)
    for thresh, style in [(0.80, "--"), (0.90, "-."), (0.95, ":")]:
        idx = np.searchsorted(cumulative, thresh)
        ax.axhline(thresh * 100, color="grey", linestyle=style, alpha=0.6,
                   label=f"{int(thresh*100)}% @ PC{idx+1}")
    ax.axvline(N_COMPONENTS_3, color="#4CAF50", linewidth=2,
               label=f"3-PC AE match ({cumulative[2]*100:.1f}%)")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Cumulative")
    ax.legend(fontsize=9)
    ax.set_xlim(1, min(50, len(cumulative)))

    plt.tight_layout()
    path = FIG_DIR / "01_scree_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
    return cumulative


# ── 5.2  PCA 2-D regime scatter ─────────────────────────────────────────────

def fig_2d_scatter(df, title_prefix, x_col, y_col, regime_col, filename):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(f"{title_prefix} — 2-D Latent Space by Regime", fontsize=14, fontweight="bold")

    for ax, split in zip(axes, ("train", "val", "test")):
        sub = df[df["split"] == split]
        for r in range(N_CLUSTERS):
            mask = sub[regime_col] == r
            ax.scatter(sub.loc[mask, x_col], sub.loc[mask, y_col],
                       c=PALETTE[r], label=f"Regime {r}", alpha=0.55, s=18, edgecolors="none")
        ax.set_title(f"{split.capitalize()} split")
        ax.set_xlabel(x_col.upper())
        if split == "train":
            ax.set_ylabel(y_col.upper())
        ax.legend(markerscale=1.4, fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── 5.3  Side-by-side 3-D projections ───────────────────────────────────────

def fig_3d_comparison(merged_df):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("PCA vs. Autoencoder — 3-D Latent Space (full dataset)",
                 fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    for r in range(N_CLUSTERS):
        mask = merged_df["pca_regime"] == r
        ax1.scatter(merged_df.loc[mask, "pc1"],
                    merged_df.loc[mask, "pc2"],
                    merged_df.loc[mask, "pc3"],
                    c=PALETTE[r], label=f"Regime {r}", s=12, alpha=0.5)
        mask2 = merged_df["ae_regime"] == r
        ax2.scatter(merged_df.loc[mask2, "z1"],
                    merged_df.loc[mask2, "z2"],
                    merged_df.loc[mask2, "z3"],
                    c=PALETTE[r], label=f"Regime {r}", s=12, alpha=0.5)

    for ax, title, labels in [
        (ax1, "PCA (3 components)", ("PC1", "PC2", "PC3")),
        (ax2, "Autoencoder (3D latent)", ("Z1", "Z2", "Z3")),
    ]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
        ax.legend(fontsize=8, markerscale=1.5)

    plt.tight_layout()
    path = FIG_DIR / "04_pca_vs_ae_3d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── 5.4  Regime timeline ─────────────────────────────────────────────────────

def fig_regime_timeline(merged_df):
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Market Regime Timeline: PCA vs. Autoencoder", fontsize=14, fontweight="bold")

    for ax, col, title in [
        (axes[0], "pca_regime", "PCA Regimes"),
        (axes[1], "ae_regime",  "Autoencoder Regimes"),
    ]:
        df_sorted = merged_df.sort_values("date")
        for r in range(N_CLUSTERS):
            mask = df_sorted[col] == r
            dates_r = df_sorted.loc[mask, "date"]
            ax.scatter(dates_r, [r] * mask.sum(),
                       c=PALETTE[r], s=6, alpha=0.7, label=f"Regime {r}")

        # shade val / test boundaries
        val_start = merged_df.loc[merged_df["split"] == "val", "date"].min()
        test_start = merged_df.loc[merged_df["split"] == "test", "date"].min()
        ax.axvline(val_start, color="grey", linestyle="--", linewidth=1.2, alpha=0.7, label="Val start")
        ax.axvline(test_start, color="black", linestyle=":", linewidth=1.5, alpha=0.9, label="Test start")

        ax.set_title(title)
        ax.set_yticks(range(N_CLUSTERS))
        ax.set_yticklabels([f"Regime {r}" for r in range(N_CLUSTERS)])
        ax.legend(ncol=6, fontsize=8, loc="upper left")

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    path = FIG_DIR / "05_regime_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── 5.5  Feature profiles per regime ─────────────────────────────────────────

def fig_feature_profiles(splits, labels, feature_names):
    """Mean feature value per PCA regime (train only, original 30-dim)."""
    # Use the raw splits (already scaled), take the last week of each window
    # by loading splits.npz (unflattened rows aligned with window end dates).
    # We approximate by taking the last time-step of each training window.
    windows = np.load(DATA_DIR / "windows.npz")
    X_train_last = windows["train"][:, -1, :]    # (n_train, 30)
    train_labels = labels["train"]

    profiles = []
    for r in range(N_CLUSTERS):
        mask = train_labels == r
        mean_vec = X_train_last[mask].mean(axis=0)
        profiles.append(mean_vec)
    profiles = np.array(profiles)    # (4, 30)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("PCA Regime Feature Profiles (train set — last week of window)",
                 fontsize=13, fontweight="bold")

    short_names = [f.replace("_", "") for f in feature_names]

    for r, ax in enumerate(axes):
        vals = profiles[r]
        colors = [PALETTE[r] if v >= 0 else "#bdbdbd" for v in vals]
        ax.barh(short_names, vals, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Regime {r}  (n={int((train_labels==r).sum())} windows)", color=PALETTE[r])
        ax.set_xlabel("Standardised value")
        ax.invert_yaxis()

    plt.tight_layout()
    path = FIG_DIR / "06_regime_feature_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── 5.6  Regime agreement heatmap ────────────────────────────────────────────

def fig_agreement_heatmap(merged_df):
    """Cross-tabulation of PCA regime vs. AE regime."""
    crosstab = pd.crosstab(
        merged_df["pca_regime"], merged_df["ae_regime"],
        rownames=["PCA Regime"], colnames=["AE Regime"],
        normalize="index",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("PCA vs. AE Regime Agreement\n(row-normalised — % of PCA regime windows)",
                 fontsize=12, fontweight="bold")

    import matplotlib.cm as cm
    im = ax.imshow(crosstab.values, cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fraction")

    ax.set_xticks(range(N_CLUSTERS))
    ax.set_xticklabels([f"AE {r}" for r in range(N_CLUSTERS)])
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_yticklabels([f"PCA {r}" for r in range(N_CLUSTERS)])

    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            ax.text(j, i, f"{crosstab.values[i,j]:.0%}",
                    ha="center", va="center", fontsize=11,
                    color="white" if crosstab.values[i, j] > 0.5 else "black")

    plt.tight_layout()
    path = FIG_DIR / "07_regime_agreement_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")

    ari = adjusted_rand_score(merged_df["pca_regime"], merged_df["ae_regime"])
    nmi = normalized_mutual_info_score(merged_df["pca_regime"], merged_df["ae_regime"])
    return ari, nmi


# ── 5.7  Variance explained vs. AE reconstruction loss summary ───────────────

def fig_summary_metrics(cumulative, ae_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Method Comparison: PCA (linear) vs. Autoencoder (non-linear)",
                 fontsize=13, fontweight="bold")

    # Left: cumulative variance
    ax = axes[0]
    ax.plot(range(1, min(51, len(cumulative)+1)), cumulative[:50] * 100,
            "o-", color="#2196F3", linewidth=2, markersize=4)
    ax.axvline(N_COMPONENTS_3, color="#FF5722", linewidth=2,
               label=f"3 components → {cumulative[2]*100:.1f}% variance")
    ax.fill_between(range(1, 4), 0, cumulative[:3] * 100, alpha=0.15, color="#FF5722")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA: Information Retention")
    ax.legend()

    # Right: reconstruction loss by split from AE
    try:
        loss_df = pd.read_csv(AE_DIR / "split_reconstruction_loss.csv")
        splits_order = loss_df["split"].tolist() if "split" in loss_df.columns else []
        losses = loss_df["reconstruction_loss"].tolist() if "reconstruction_loss" in loss_df.columns else []
        colors_bar = [SPLIT_COLORS.get(s, "#888") for s in splits_order]
        ax2 = axes[1]
        bars = ax2.bar(splits_order, losses, color=colors_bar, edgecolor="white", width=0.5)
        for bar, val in zip(bars, losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontsize=10)
        ax2.set_ylabel("MSE Reconstruction Loss")
        ax2.set_title("AE: Reconstruction Loss by Split")
    except Exception:
        axes[1].text(0.5, 0.5, "split_reconstruction_loss.csv\nnot found",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    path = FIG_DIR / "08_reconstruction_loss_vs_pca_variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. INTERPRETATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

REGIME_LABELS = {
    0: "Regime 0",
    1: "Regime 1",
    2: "Regime 2",
    3: "Regime 3",
}


def build_interpretation(merged_df, cumulative, ari, nmi, labels, splits):
    """Print a structured interpretation of the PCA regimes."""
    windows = np.load(DATA_DIR / "windows.npz")
    X_train_last = windows["train"][:, -1, :]
    with open(DATA_DIR / "schema.json") as f:
        feature_names = json.load(f)["feature_order"]

    train_labels = labels["train"]
    print("\n" + "=" * 72)
    print("  PCA MARKET REGIME ANALYSIS — INTERPRETATION REPORT")
    print("=" * 72)

    print(f"\n[1] Dimensionality Reduction (PCA)")
    print(f"    3 principal components explain {cumulative[2]*100:.1f}% of variance")
    print(f"    10 components: {cumulative[9]*100:.1f}%  |  "
          f"20 components: {cumulative[19]*100:.1f}%  |  "
          f"50 components: {cumulative[min(49, len(cumulative)-1)]*100:.1f}%")
    print(f"    → Substantial information is lost in 3-D PCA; the AE compresses")
    print(f"      non-linearly and may preserve structure PCA misses.")

    print(f"\n[2] Regime Sizes (train set, {N_CLUSTERS} clusters via KMeans on PCA space)")
    for r in range(N_CLUSTERS):
        n = int((train_labels == r).sum())
        pct = n / len(train_labels) * 100
        print(f"    Regime {r}: {n:4d} windows ({pct:.1f}%)")

    print(f"\n[3] Feature Signatures (mean standardised value — top 3 per regime)")
    for r in range(N_CLUSTERS):
        mask = train_labels == r
        mean_vec = X_train_last[mask].mean(axis=0)
        top_idx = np.argsort(np.abs(mean_vec))[::-1][:5]
        sig = ", ".join(
            f"{feature_names[i]}={mean_vec[i]:+.2f}" for i in top_idx
        )
        print(f"    Regime {r}: {sig}")

    print(f"\n[4] PCA ↔ Autoencoder Agreement (full dataset)")
    print(f"    Adjusted Rand Index (ARI): {ari:.3f}  (1=perfect, 0=random)")
    print(f"    Normalized Mutual Info   : {nmi:.3f}  (1=perfect, 0=none)")
    if ari > 0.6:
        print("    → Strong agreement: both methods identify similar market structures.")
    elif ari > 0.3:
        print("    → Moderate agreement: methods share some regime structure but differ")
        print("      in boundary placement, reflecting AE's non-linear capacity.")
    else:
        print("    → Low agreement: AE captures non-linear regime structure that linear")
        print("      PCA cannot reproduce with only 3 components.")

    print(f"\n[5] Regime Temporal Interpretation")
    df_s = merged_df.sort_values("date")
    for r in range(N_CLUSTERS):
        mask = df_s["pca_regime"] == r
        dates_r = df_s.loc[mask, "date"]
        if len(dates_r):
            print(f"    Regime {r}: {dates_r.min().strftime('%Y-%m')} – "
                  f"{dates_r.max().strftime('%Y-%m')}  "
                  f"({mask.sum()} windows)")

    print("\n[6] Suggested Regime Interpretations")
    print("    (Based on feature profiles; requires domain validation)")

    for r in range(N_CLUSTERS):
        mask = train_labels == r
        mean_vec = X_train_last[mask].mean(axis=0)
        feat_dict = dict(zip(feature_names, mean_vec))

        descriptors = []
        if feat_dict.get("_MKT", 0) > 0.3:
            descriptors.append("bull equity market")
        elif feat_dict.get("_MKT", 0) < -0.3:
            descriptors.append("bear equity market")
        if feat_dict.get("MOV", 0) > 0.3:
            descriptors.append("high volatility")
        elif feat_dict.get("MOV", 0) < -0.3:
            descriptors.append("low volatility")
        if feat_dict.get("UN", 0) > 0.3:
            descriptors.append("elevated unemployment")
        if feat_dict.get("CPI", 0) > 0.3:
            descriptors.append("inflationary pressure")
        if feat_dict.get("Y10", 0) > 0.3:
            descriptors.append("rising long rates")
        elif feat_dict.get("Y10", 0) < -0.3:
            descriptors.append("falling long rates")
        if feat_dict.get("_OIL", 0) > 0.3:
            descriptors.append("oil price strength")
        if not descriptors:
            descriptors.append("mixed / transitional")

        print(f"    Regime {r}: {', '.join(descriptors)}")

    print("\n" + "=" * 72 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data …")
    splits, dates, feature_names = load_windows()
    ae_df = load_ae_results()

    print("Fitting PCA …")
    full_pca, pca_3, projections = fit_pca(splits)

    print("Clustering PCA latent space …")
    km, labels = cluster_pca(projections)

    print("Building output dataframes …")
    pca_df = build_dataframes(projections, labels, dates)
    merged_df = merge_with_ae(pca_df, ae_df)

    # Save CSVs
    latent_out = pca_df[["split", "window_index", "date", "pc1", "pc2", "pc3"]]
    latent_out.to_csv(OUT_DIR / "pca_latent_space.csv", index=False)
    pca_df.to_csv(OUT_DIR / "pca_clustered_regimes.csv", index=False)
    merged_df.to_csv(OUT_DIR / "pca_ae_comparison.csv", index=False)
    print("  Saved pca_latent_space.csv, pca_clustered_regimes.csv, pca_ae_comparison.csv")

    print("\nGenerating figures …")
    cumulative = fig_scree(full_pca)
    fig_2d_scatter(pca_df, "PCA", "pc1", "pc2", "regime",
                   "02_pca_2d_regimes.png")
    fig_2d_scatter(ae_df, "Autoencoder", "z1", "z2", "regime",
                   "03_ae_2d_regimes.png")
    fig_3d_comparison(merged_df)
    fig_regime_timeline(merged_df)
    fig_feature_profiles(splits, labels, feature_names)
    ari, nmi = fig_agreement_heatmap(merged_df)
    fig_summary_metrics(cumulative, ae_df)

    build_interpretation(merged_df, cumulative, ari, nmi, labels, splits)

    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
