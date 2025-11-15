import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spielerstats
import pässengegenblock
from pässengegenblock import predict_block_by_passes
from binomialverteilung_ultis import binomial_expectation

# Reload modules
importlib.reload(spielerstats)
importlib.reload(pässengegenblock)


# 1) Treffer Wahrscheinlichkeit

def get_player_hit_probs(player_name: str, shot_type: str):
    """
    Returns (p_nb, p_b):
      p_nb = hit probability without block
      p_b  = hit probability with block
    Based on the cleaned spielerstats.df.
    """
    df = spielerstats.df

    p = player_name.strip().lower()
    st = shot_type.strip().lower()

    player_df = df[df["player_name"].str.lower() == p]
    if player_df.empty:
        return 0.0, 0.0

    player_df = player_df[player_df["shot_type"] == st]
    if player_df.empty:
        return 0.0, 0.0

    nb_df = player_df[player_df["block"] == 0]
    b_df  = player_df[player_df["block"] == 1]

    p_nb = float(nb_df["hit"].mean()) if len(nb_df) > 0 else 0.0
    p_b  = float(b_df["hit"].mean()) if len(b_df) > 0 else 0.0

    return p_nb, p_b



# 2) Teams bestimmen

def build_teams_from_players(team1_name: str, team2_name: str, player_specs):
    """
    Zwei Teams werden durch eine Liste bestimmt: player_specs.
    player_specs ist eine Liste von Tuplen:
      (player_name, team_name, total_attempts, n_wurf, n_3er, n_layup, passes)
    """
    team1 = {"name": team1_name, "players": []}
    team2 = {"name": team2_name, "players": []}

    attempts_team1 = 0
    attempts_team2 = 0

    for spec in player_specs:
        (
            player_name,
            t_name,
            total_attempts,
            n_wurf,
            n_3er,
            n_layup,
            passes,
        ) = spec

        # Konsistenz Debugging
        if total_attempts != (n_wurf + n_3er + n_layup):
            raise ValueError(
                f"Attempt mismatch for player {player_name}: "
                f"total={total_attempts}, breakdown={n_wurf + n_3er + n_layup}"
            )

        # Team der Spieler bestimmen
        if t_name == team1_name:
            team = team1
            attempts_team1 += total_attempts
        elif t_name == team2_name:
            team = team2
            attempts_team2 += total_attempts
        else:
            raise ValueError(
                f"Invalid team name '{t_name}' for player {player_name}. "
                f"Allowed: '{team1_name}' or '{team2_name}'"
            )

        # Anzahl jeder Wurfart hinzufügen
        if n_wurf > 0:
            team["players"].append(
                {"name": player_name, "shot_type": "wurf", "passes": passes, "attempts": int(n_wurf)}
            )
        if n_3er > 0:
            team["players"].append(
                {"name": player_name, "shot_type": "3er-wurf", "passes": passes, "attempts": int(n_3er)}
            )
        if n_layup > 0:
            team["players"].append(
                {"name": player_name, "shot_type": "layup", "passes": passes, "attempts": int(n_layup)}
            )

    # Versuch Anzahl Debugging
    if attempts_team1 > 100:
        raise ValueError(f"Team '{team1_name}' > 100 attempts ({attempts_team1})")
    if attempts_team2 > 100:
        raise ValueError(f"Team '{team2_name}' > 100 attempts ({attempts_team2})")

    return team1, team2



# 3) Team Leistung simulieren (Monte-Carlo)

def simulate_team_once(team: dict, threshold_block: float = 0.5, track_blocks: bool = False):
    """
    Simuliert alle Versuche für ein Team (Monte-Carlo).
    Returns:
      total_points, total_blocks, details
    details = Liste von Dicts mit Infos pro Spieler und Wurfart.
    """
    total_points = 0
    total_blocks = 0
    details = []

    team_name = team.get("name", "Team")

    for cfg in team["players"]:
        name = cfg["name"]
        shot_type = cfg["shot_type"]
        passes = cfg["passes"]
        n = int(cfg["attempts"])

        p_nb, p_b = get_player_hit_probs(name, shot_type)
        _, p_block = predict_block_by_passes(passes, threshold=threshold_block)

        # Block Simulation
        block_events = np.random.rand(n) < p_block
        n_blocked = int(block_events.sum())
        n_not_blocked = n - n_blocked

        if track_blocks:
            total_blocks += n_blocked

        # Simulation der Versuche unter Block
        if n_blocked > 0 and p_b > 0:
            hits_block = np.random.rand(n_blocked) < p_b
        else:
            hits_block = np.zeros(n_blocked, dtype=bool)

        # Simulation der Versuche ohne Block
        if n_not_blocked > 0 and p_nb > 0:
            hits_nb = np.random.rand(n_not_blocked) < p_nb
        else:
            hits_nb = np.zeros(n_not_blocked, dtype=bool)

        # Punkt Bestimmung je nach Wurfart
        if shot_type == "3er-wurf":
            points_per_hit = 3
        else:  # layup oder normaler wurf = 2 Punkte
            points_per_hit = 2

        # Punkten summieren
        total_hits = int(hits_block.sum() + hits_nb.sum())
        points = total_hits * points_per_hit
        total_points += points

        # Details für diese Spieler+Wurfart-Kombi speichern
        details.append({
            "team": team_name,
            "player": name,
            "shot_type": shot_type,
            "attempts": n,
            "hits": total_hits,
            "points": points,
            "blocks": n_blocked,
        })

    return total_points, total_blocks, details



# 4) Erwartete Punkte pro Team (Theorie, ohne Monte-Carlo)

def expected_points_team(team: dict, threshold_block: float = 0.5) -> float:
    """
    Berechnet den theoretischen Erwartungswert der Punkte eines Teams.
    Idee:
      - p_block = P(Block | passes)
      - p_nb    = P(Treffer | kein Block)
      - p_b     = P(Treffer | Block)

      effektive Treffer-Wahrscheinlichkeit:
        p_hit_eff = (1 - p_block) * p_nb + p_block * p_b

      Trefferzahl ~ Binomial(n, p_hit_eff)
      -> E[Hits] = n * p_hit_eff
      -> E[Punkte] = E[Hits] * points_per_hit
    """
    total_E_points = 0.0

    for cfg in team["players"]:
        name = cfg["name"]
        shot_type = cfg["shot_type"]
        passes = cfg["passes"]
        n = int(cfg["attempts"])

        p_nb, p_b = get_player_hit_probs(name, shot_type)
        _, p_block = predict_block_by_passes(passes, threshold=threshold_block)

        # effektive Treffer-Wahrscheinlichkeit (Block + kein Block kombiniert)
        p_hit_eff = (1.0 - p_block) * p_nb + p_block * p_b

        # Binomial Erwartungswert der Treffer
        expected_hits = binomial_expectation(n, p_hit_eff)

        # Punkte pro Treffer
        if shot_type == "3er-wurf":
            points_per_hit = 3
        else:
            points_per_hit = 2

        total_E_points += expected_hits * points_per_hit

    return total_E_points



# 4b) Erwartete Details pro Spieler und Wurfart (Theorie)

def expected_details_team(team: dict, threshold_block: float = 0.5):
    """
    Erwartete Werte pro Spieler und Wurfart für ein Team.
    Gibt eine Liste von Dicts zurück, analog zu den Simulation-Details:
      - team, player, shot_type, attempts
      - expected_hits, expected_points
    """
    details = []
    team_name = team.get("name", "Team")

    for cfg in team["players"]:
        name = cfg["name"]
        shot_type = cfg["shot_type"]
        passes = cfg["passes"]
        n = int(cfg["attempts"])

        p_nb, p_b = get_player_hit_probs(name, shot_type)
        _, p_block = predict_block_by_passes(passes, threshold=threshold_block)

        # effektive Treffer-Wahrscheinlichkeit
        p_hit_eff = (1.0 - p_block) * p_nb + p_block * p_b

        # Erwartete Treffer und Punkte
        expected_hits = binomial_expectation(n, p_hit_eff)

        if shot_type == "3er-wurf":
            points_per_hit = 3
        else:
            points_per_hit = 2

        expected_points = expected_hits * points_per_hit

        details.append({
            "team": team_name,
            "player": name,
            "shot_type": shot_type,
            "attempts": n,
            "expected_hits": expected_hits,
            "expected_points": expected_points,
            "p_hit_eff": p_hit_eff,
        })

    return details



# 5) Eine Simulation: Teams bauen + simulieren + Gewinner bestimmen

def simulate_match_from_player_specs(
    team1_name: str,
    team2_name: str,
    player_specs,
    threshold_block: float = 0.5,
):
    """
    Nimmt player_specs, baut daraus zwei Teams,
    simuliert EIN Spiel (Monte-Carlo) und gibt das Ergebnis zurück.
    """

    # Teams aus der Spezifikation bauen
    team1, team2 = build_teams_from_players(team1_name, team2_name, player_specs)

    # Spiel simulieren
    score1, blocks1, details1 = simulate_team_once(team1, threshold_block, track_blocks=True)
    score2, blocks2, details2 = simulate_team_once(team2, threshold_block, track_blocks=True)

    # Gewinner bestimmen
    if score1 > score2:
        winner = team1_name
    elif score2 > score1:
        winner = team2_name
    else:
        winner = "Draw"

    # alle Detaildaten zusammen (für Plots)
    all_details = details1 + details2

    return {
        "team1_name": team1_name,
        "team2_name": team2_name,
        f"{team1_name} points": score1,
        f"{team2_name} points": score2,
        f"{team1_name} shots blocked": blocks1,
        f"{team2_name} shots blocked": blocks2,
        "winner": winner,
        "details": all_details,
    }



# 6) Erwartetes Ergebnis für beide Teams (ohne Simulation)

def expected_match_from_player_specs(
    team1_name: str,
    team2_name: str,
    player_specs,
    threshold_block: float = 0.5,
):
    """
    Nimmt die gleiche player_specs wie die Simulation,
    berechnet aber nur die erwarteten Punkte pro Team (keine Zufallsziehung).
    """
    team1, team2 = build_teams_from_players(team1_name, team2_name, player_specs)

    exp1 = expected_points_team(team1, threshold_block=threshold_block)
    exp2 = expected_points_team(team2, threshold_block=threshold_block)

    if exp1 > exp2:
        better = team1_name
    elif exp2 > exp1:
        better = team2_name
    else:
        better = "Equal"

    return {
        "team1_name": team1_name,
        "team2_name": team2_name,
        f"{team1_name} expected score": exp1,
        f"{team2_name} expected score": exp2,
        "better team by expectation": better,
    }



# 6b) Erwartete Details für beide Teams (Spieler + Wurfart)

def expected_details_from_player_specs(
    team1_name: str,
    team2_name: str,
    player_specs,
    threshold_block: float = 0.5,
):
    """
    Baut die Teams und gibt erwartete Detailwerte
    (pro Spieler und Wurfart) für beide Teams zurück.
    """
    team1, team2 = build_teams_from_players(team1_name, team2_name, player_specs)

    details1 = expected_details_team(team1, threshold_block=threshold_block)
    details2 = expected_details_team(team2, threshold_block=threshold_block)

    all_details = details1 + details2

    return {
        "team1_name": team1_name,
        "team2_name": team2_name,
        "details": all_details,
    }



# 7) Balkendiagramme für eine Simulation (Monte-Carlo)

def plot_match_barcharts(result: dict):
    """
    Nimmt das Ergebnis von simulate_match_from_player_specs
    und zeichnet:
      - Punkte pro Spieler und Wurfart (gestapelte Balken)
      - Gesamtpunkte pro Team
    """
    details = result.get("details", [])
    if not details:
        print("Keine Details zum Plotten vorhanden (result['details'] fehlt).")
        return

    df = pd.DataFrame(details)

    # Punkte pro Spieler und Wurfart (Stacked Bar)
    pivot = df.pivot_table(
        index="player",
        columns="shot_type",
        values="points",
        aggfunc="sum",
        fill_value=0,
    )

    ax = pivot.plot(kind="bar", stacked=True)
    ax.set_ylabel("Punkte")
    ax.set_title("Punkte pro Spieler und Wurfart (Simulation)")
    ax.set_xlabel("Spieler")
    ax.legend(title="Wurfart")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Gesamtpunkte pro Team
    team_points = df.groupby("team")["points"].sum()

    ax2 = team_points.plot(kind="bar")
    ax2.set_ylabel("Punkte")
    ax2.set_title("Team-Gesamtpunkte (Simulation)")
    ax2.set_xlabel("Team")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# 9) Balkendiagramm für erwartete Punkte pro Spieler und Wurfart

def plot_expected_details_barcharts(expected_result: dict):
    """
    Nimmt das Ergebnis von expected_details_from_player_specs
    und zeichnet:
      - erwartete Punkte pro Spieler und Wurfart (gestapelte Balken)
      - erwartete Gesamtpunkte pro Team
    """
    details = expected_result.get("details", [])
    if not details:
        print("Keine erwarteten Details zum Plotten vorhanden.")
        return

    df = pd.DataFrame(details)

    # Erwartete Punkte pro Spieler und Wurfart
    pivot = df.pivot_table(
        index="player",
        columns="shot_type",
        values="expected_points",
        aggfunc="sum",
        fill_value=0,
    )

    ax = pivot.plot(kind="bar", stacked=True)
    ax.set_ylabel("Erwartete Punkte")
    ax.set_title("Erwartete Punkte pro Spieler und Wurfart")
    ax.set_xlabel("Spieler")
    ax.legend(title="Wurfart")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Erwartete Gesamtpunkte pro Team
    team_points = df.groupby("team")["expected_points"].sum()

    ax2 = team_points.plot(kind="bar")
    ax2.set_ylabel("Erwartete Punkte")
    ax2.set_title("Erwartete Team-Gesamtpunkte")
    ax2.set_xlabel("Team")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
