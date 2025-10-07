import pandas as pd
import matplotlib.pyplot as plt

# --- Load and clean data ---
df = pd.read_csv("Basketball_Daten.csv", sep=";", encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.lower()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# --- Convert categories to numeric ---
df["block"] = df["block"].map({"Ja": 1, "Nein": 0})
df["hit"] = df["hit"].map({"Ja": 1, "Nein": 0})

# --- Group by number of passes ---
block_stats = df.groupby("passes")["block"].agg(
    total_wuerfe="count",
    blocked_sum="sum"
).reset_index()

print(block_stats)

# --- Plot: total vs blocked shots ---
plt.figure(figsize=(8, 5))
plt.bar(block_stats["passes"], block_stats["total_wuerfe"], color="#99CCFF", label="Gesamtwürfe")
plt.bar(block_stats["passes"], block_stats["blocked_sum"], color="#3366FF", label="Geblockte Würfe")

plt.xlabel("Anzahl Pässe vor Wurf")
plt.ylabel("Anzahl Würfe")
plt.title("Geblockte vs. gesamte Würfe nach Passanzahl")
plt.legend()
plt.tight_layout()
plt.show()
