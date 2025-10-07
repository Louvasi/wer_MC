import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Dataframe erstellen
df = pd.read_csv("Basketball_Daten.csv", sep=";", encoding="utf-8-sig")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# --- Trim all string values in every column ---
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Now convert 'hit' and Block
df["hit"] = df["hit"].map({"Ja": 1, "Nein": 0})
df["block"] = df["block"].map({"Ja": 1, "Nein": 0})

table = df.groupby(["player_name", "block"])["hit"].value_counts().unstack(fill_value=0)
print(table)

 # Group data
hit_summary = df.groupby(["player_name", "block"])["hit"].value_counts().unstack(fill_value=0)

# Reset index and pivot so all players appear for both block values
plot_df = hit_summary.reset_index()
plot_df = plot_df.pivot(index="player_name", columns="block", values=1).fillna(0)

# Prepare data
players = plot_df.index
block0_hits = plot_df[0]
block1_hits = plot_df[1] if 1 in plot_df.columns else np.zeros(len(players))

x_indexes = np.arange(len(players))
bar_width = 0.35

# Create bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x_indexes - bar_width/2, block0_hits, width=bar_width, label="Block = 0 (No Block)")
ax.bar(x_indexes + bar_width/2, block1_hits, width=bar_width, label="Block = 1 (Blocked)")

# Customize chart
ax.set_xlabel("Player")
ax.set_ylabel("Number of Hits")
ax.set_title("Hits per Player by Block Status")
ax.set_xticks(x_indexes)
ax.set_xticklabels(players)
ax.legend()

plt.tight_layout()
plt.show()