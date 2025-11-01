import pandas as pd

# --- Load & clean data ---
CSV_PATH = r"C:\Users\lukas\Desktop\gpr\wer_MC\Basketball_Daten.csv"

df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")

# Normalize columns and string values
df.columns = df.columns.str.strip().str.lower()   # fixes "shot_type " -> "shot_type"
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Map German labels to numeric for hits/blocks
if df["hit"].dtype == object:
    df["hit"] = df["hit"].map({"Ja": 1, "Nein": 0}).astype("Int64")
if df["block"].dtype == object:
    df["block"] = df["block"].map({"Ja": 1, "Nein": 0}).astype("Int64")

# Make shot_type consistent
df["shot_type"] = df["shot_type"].str.lower()  # values: "wurf", "layup", "3er-wurf"

def _rate_for_type(player_df: pd.DataFrame, shot_type_value: str) -> float:
    """Overall hit rate for a given shot type within the player's rows."""
    subset = player_df[player_df["shot_type"] == shot_type_value]
    if len(subset) == 0:
        return 0.0
    return float(round(subset["hit"].mean(), 3))

def _rate_for_type_and_block(player_df: pd.DataFrame, shot_type_value: str, block_value: int) -> float:
    """Hit rate for a given shot type under a specific block condition."""
    subset = player_df[
        (player_df["shot_type"] == shot_type_value) &
        (player_df["block"] == block_value)
    ]
    if len(subset) == 0:
        return 0.0
    return float(round(subset["hit"].mean(), 3))

def get_player_stats(player: str) -> str:
    """
    Returns a formatted string with:
      - Spieler
      - Gesamt Würfe
      - Erfolg-Rate je Wurfart with labeled p_w, p_lp, p_3w (overall)
      - Split 'ohne Block' (Block=0) and 'mit Block' (Block=1) for each Wurfart
    """
    p = player.strip().lower()
    player_df = df[df["player_name"].str.lower() == p]

    if player_df.empty:
        return f"Keine Daten für Spieler {player}"

    total = int(len(player_df))

    # Shot types present in the data
    types = [
        ("wurf", "Wurf", "p_w"),
        ("layup", "Lay-up", "p_lp"),
        ("3er-wurf", "3-Wurf", "p_3w"),
    ]

    lines = [f"Spieler: {player}", f"Gesamt Würfe: {total}"]

    for key, label, sym in types:
        p_overall = _rate_for_type(player_df, key)
        p_nb = _rate_for_type_and_block(player_df, key, 0)  # ohne Block
        p_b  = _rate_for_type_and_block(player_df, key, 1)  # mit Block
        lines.append(
            f"Erfolg-Rate {label} {sym} = {p_overall:.2f} "
            f"(ohne Block = {p_nb:.2f}, mit Block = {p_b:.2f})"
        )

    return "\n".join(lines)

"""
Example Usage 
print(get_player_stats("Alexis"))
print(get_player_stats("Jakov"))
print(get_player_stats("Loukas"))
"""