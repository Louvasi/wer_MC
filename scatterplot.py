import pandas as pd
import matplotlib.pyplot as plt

# Daten laden
df = pd.read_csv("Basketball_Daten.csv", sep=";", skipinitialspace=True)

# Punkte nach Passanzahl summieren
summary = df.groupby("passes")["points"].sum().reset_index()

print(summary)  # Kontrolle in der Konsole

# Scatterplot erstellen
plt.figure(figsize=(7, 6))
plt.scatter(summary["passes"], summary["points"], s=120, color="orange", edgecolors="black")

plt.xlabel("Pässe")
plt.ylabel("Gesamtpunkte")
plt.title("Gesamtpunkte nach Anzahl der Pässe")

plt.xticks(range(0, 7))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
