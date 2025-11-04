# fisher_hypothesentest_minimal.py
from scipy.stats import fisher_exact
from spielerstats import df

ALPHA = 0.05

def _normalize_shot_type(s: str) -> str:
    s = s.strip().lower()
    mapping = {
        "wurf": "wurf",
        "layup": "layup",
        "lay-up": "layup",
        "lay up": "layup",
        "3er-wurf": "3er-wurf",
        "3-wurf": "3er-wurf",
        "3er": "3er-wurf",
        "dreier": "3er-wurf",
        "3": "3er-wurf",
    }
    return mapping.get(s, s)

def fisher_hypothesentest(player: str, shot_type: str) -> str:
    st = _normalize_shot_type(shot_type)
    p_name = player.strip().lower()

    player_df = df[df["player_name"].str.lower() == p_name]
    sub = player_df[player_df["shot_type"].str.lower() == st]

    if player_df.empty or sub.empty:
        return f"In {shot_type} der {player} hat p_Wert = —: kein Hinweis auf Block-Effekt"

    nb = sub[sub["block"] == 0]
    b  = sub[sub["block"] == 1]

    a = int(nb["hit"].sum())
    b0 = int((1 - nb["hit"]).sum())
    c = int(b["hit"].sum())
    d = int((1 - b["hit"]).sum())

    # Falls keine Blockwürfe existieren → neutraler Satz
    if b.shape[0] == 0:
        return f"In {shot_type} der {player} hat p_Wert = —: kein Hinweis auf Block-Effekt"

    _, p_value = fisher_exact([[a, b0],[c, d]], alternative="less")

    if p_value < 0.01:
        level = "hochsignifikante Evidenz für Block-Effekt"
    elif p_value < 0.05:
        level = "signifikante Evidenz für Block-Effekt"
    elif p_value < 0.10:
        level = "Trend zu Block-Effekt"
    else:
        level = "kein Hinweis auf Block-Effekt"

    return f"In {shot_type} der {player} hat p_Wert = {p_value:.4g}: {level}"
