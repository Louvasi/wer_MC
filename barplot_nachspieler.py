import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dataframe erstellen
df = pd.read_csv("Basketball_Daten.csv", sep=";", encoding="utf-8-sig")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Trim all string values in every column
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Now convert 'hit' and Block
df["hit"] = df["hit"].map({"Ja": 1, "Nein": 0})
df["block"] = df["block"].map({"Ja": 1, "Nein": 0})

table = df.groupby(["player_name", "block"])["hit"].value_counts().unstack(fill_value=0)
print(table)

# --- Group data: count hits and misses by player and block ---
hit_summary = df.groupby(["player_name", "block"])["hit"].value_counts().unstack(fill_value=0)

# --- Ensure all players have both block values ---
hit_summary = hit_summary.unstack(level="block", fill_value=0)

# --- Extract counts ---
miss_block0 = hit_summary[(0, 0)]
miss_block1 = hit_summary[(0, 1)]
hit_block0  = hit_summary[(1, 0)]
hit_block1  = hit_summary[(1, 1)]

players = hit_summary.index
x = np.arange(len(players))
bar_width = 0.35

# --- Create bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))

# Misses (stacked)
ax.bar(x - bar_width/2, miss_block0, width=bar_width, color="#FF9999", label="Niete - Block 0")
ax.bar(x - bar_width/2, miss_block1, width=bar_width, bottom=miss_block0, color="#FF4C4C", label="Niete - Block 1")

# Hits (stacked)
ax.bar(x + bar_width/2, hit_block0, width=bar_width, color="mediumseagreen", label="Treffer - Block 0")
ax.bar(x + bar_width/2, hit_block1, width=bar_width, bottom=hit_block0, color="lime", label="Treffer - Block 1")

# --- Customize chart ---
ax.set_xlabel("Spieler")
ax.set_ylabel("Anzahl Würfe")
ax.set_title("Treffer und Niete pro Spieler (nach Blockstatus gestapelt)")
ax.set_xticks(x)
ax.set_xticklabels(players)
ax.legend(ncol=2, loc="upper left")

plt.tight_layout()
plt.show()