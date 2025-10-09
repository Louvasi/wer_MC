import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load and clean data ---
df = pd.read_csv("Basketball_Daten.csv", sep=";", encoding="utf-8-sig")

df.columns = df.columns.str.strip().str.lower()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert categories to numbers
df["hit"] = df["hit"].map({"Ja": 1, "Nein": 0})
df["block"] = df["block"].map({"Ja": 1, "Nein": 0})

# --- Get all unique shot types ---
shot_types = df["shot_type"].unique()

# --- Create subplots horizontally ---
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(shot_types),
    figsize=(8 * len(shot_types), 6),
    sharey=True
)

# Make sure axes is iterable
if len(shot_types) == 1:
    axes = [axes]

# --- Loop over each shot type ---
for ax, shot_type in zip(axes, shot_types):
    # Filter for this shot type
    subset = df[df["shot_type"] == shot_type]

    # Group data: hits/misses by player and block
    hit_summary = subset.groupby(["player_name", "block"])["hit"].value_counts().unstack(fill_value=0)
    hit_summary = hit_summary.unstack(level="block", fill_value=0)

    # Extract counts safely
    miss_block0 = hit_summary[(0, 0)] if (0, 0) in hit_summary.columns else np.zeros(len(hit_summary))
    miss_block1 = hit_summary[(0, 1)] if (0, 1) in hit_summary.columns else np.zeros(len(hit_summary))
    hit_block0  = hit_summary[(1, 0)] if (1, 0) in hit_summary.columns else np.zeros(len(hit_summary))
    hit_block1  = hit_summary[(1, 1)] if (1, 1) in hit_summary.columns else np.zeros(len(hit_summary))

    players = hit_summary.index
    x = np.arange(len(players))
    bar_width = 0.35

    # --- Stacked bars ---
    ax.bar(x - bar_width/2, miss_block0, width=bar_width, color="#FF9999", label="Niete - Block 0")
    ax.bar(x - bar_width/2, miss_block1, width=bar_width, bottom=miss_block0, color="#FF4C4C", label="Niete - Block 1")
    ax.bar(x + bar_width/2, hit_block0, width=bar_width, color="mediumseagreen", label="Treffer - Block 0")
    ax.bar(x + bar_width/2, hit_block1, width=bar_width, bottom=hit_block0, color="lime", label="Treffer - Block 1")

    # --- Labels and title ---
    ax.set_xlabel("Spieler")
    ax.set_ylabel("Anzahl Würfe")
    ax.set_title(f"Wurfart: {shot_type}")
    ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=20)
    ax.legend(ncol=1, loc="upper left")

# --- Adjust layout ---
plt.tight_layout()
plt.show()
