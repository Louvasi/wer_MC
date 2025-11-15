import pandas as pd

# 1. CSV korrekt einlesen (Trennzeichen ; )
df = pd.read_csv("Basketball_Daten.csv", sep=";")

# 2. Spaltennamen von Leerzeichen bereinigen
df.columns = df.columns.str.strip()
# Jetzt heißen die Spalten: id, player_name, shot_type, block, passes, points, hit

# 3. Block-Flag aus "Ja"/"Nein" erzeugen
df["block_flag"] = df["block"].map({"Ja": 1, "Nein": 0})

# 4. Blockwahrscheinlichkeit pro Passanzahl berechnen
pass_stats = (
    df.groupby("passes")["block_flag"]
      .mean()
      .reset_index()
      .rename(columns={"block_flag": "block_prob"})
)

# 5. Lookup-Table und Gesamtwahrscheinlichkeit
prob_by_passes = dict(zip(pass_stats["passes"], pass_stats["block_prob"]))
overall_prob = df["block_flag"].mean()


def predict_block_by_passes(passes, threshold=0.5):
    prob = prob_by_passes.get(passes, overall_prob)

    # Cast to normal Python types
    prob = float(prob)
    blocked = bool(prob >= threshold)

    return blocked, prob


# 6. Kurzer Test
if __name__ == "__main__":
    for k in sorted(df["passes"].unique()):
        blocked, p = predict_block_by_passes(k)
        print(f"Pässe={k}: p_block={p:.3f}, Vorhersage={'BLOCK' if blocked else 'KEIN BLOCK'}")

