"""
Recupera alpha, beta, gamma, rho desde el template v3 (forescore_mundial_dashboard_final.html)
por LSQ sobre las 144 lambdas observadas (72 partidos × 2).

Reconstruye también:
- penalty_probs.csv (probabilidades históricas de ganar tanda de penales)
- draw_tendency.csv (desviación empírica respecto al empate Poisson)
- optimized_params.csv (thresholds calibrados para predicción 1X2)
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path

WORKDIR = Path(__file__).parent
HTML = WORKDIR / "dashboard_template.html"

text = HTML.read_text()
m_teams = re.search(r"const teams = (\[.*?\]);", text, re.DOTALL)
m_matches = re.search(r"const matches = (\[.*?\]);", text, re.DOTALL)
teams = json.loads(m_teams.group(1))
matches = json.loads(m_matches.group(1))

# Mapeo es <-> en
en_by_es = {t["team_es"]: t["team"] for t in teams}
es_by_en = {t["team"]: t["team_es"] for t in teams}

# Los matches usan team_es; convertimos a en
team_names = sorted({t["team"] for t in teams})
n = len(team_names)
idx = {t: i for i, t in enumerate(team_names)}

# Sistema lineal: log(lh) = alpha_h - beta_a + gamma ; log(la) = alpha_a - beta_h
# Vars: alpha[0..n-1], beta[n..2n-1], gamma=2n
GAMMA = 2 * n
rows, y = [], []

for mt in matches:
    h_en = en_by_es[mt["home_es"]]
    a_en = en_by_es[mt["away_es"]]
    h, a = idx[h_en], idx[a_en]
    lh, la = mt["lambda_h"], mt["lambda_a"]

    row = np.zeros(2 * n + 1); row[h] = 1; row[n + a] = -1; row[GAMMA] = 1
    rows.append(row); y.append(np.log(lh))

    row = np.zeros(2 * n + 1); row[a] = 1; row[n + h] = -1
    rows.append(row); y.append(np.log(la))

# Identificabilidad
W = 1000.0
row = np.zeros(2 * n + 1); row[:n] = W; rows.append(row); y.append(0.0)
row = np.zeros(2 * n + 1); row[n:2 * n] = W; rows.append(row); y.append(0.0)

A = np.array(rows); y = np.array(y)
sol, *_ = np.linalg.lstsq(A, y, rcond=None)
alpha = sol[:n]; beta = sol[n:2 * n]; gamma = float(sol[GAMMA])

# Validar
errs = []
for mt in matches:
    h = idx[en_by_es[mt["home_es"]]]
    a = idx[en_by_es[mt["away_es"]]]
    lh_pred = np.exp(alpha[h] - beta[a] + gamma)
    la_pred = np.exp(alpha[a] - beta[h])
    errs.append(abs(lh_pred - mt["lambda_h"]))
    errs.append(abs(la_pred - mt["lambda_a"]))
errs = np.array(errs)
print(f"Error reconstrucción lambdas: mean={errs.mean():.4f}, max={errs.max():.4f}")
print(f"gamma = {gamma:.4f}")

rho = -0.092

# Guardar
pd.DataFrame({"team": team_names, "alpha": alpha, "beta": beta}).to_csv(
    WORKDIR / "dixon_coles_params.csv", index=False)
pd.DataFrame({"gamma": [gamma], "rho": [rho]}).to_csv(
    WORKDIR / "dixon_coles_global.csv", index=False)

# Guardar JSON de teams/matches para que build_dashboard los relea
with open(WORKDIR / "teams_base.json", "w") as f: json.dump(teams, f, ensure_ascii=False, indent=2)
with open(WORKDIR / "matches_base.json", "w") as f: json.dump(matches, f, ensure_ascii=False, indent=2)

# ====================================================================
# penalty_probs.csv — del reporte v3
# ====================================================================
# Valores reportados explícitamente:
penalty_reported = {
    "Germany": 0.79,
    "Argentina": 0.75,
    "Croatia": 0.75,
    "Portugal": 0.67,
    "Morocco": 0.57,    # mencionado como "histórico mediocre"
    "Spain": 0.36,
    "England": 0.36,
    "Mexico": 0.33,
    "Japan": 0.33,
}
# Resto: prior bayesiano = 0.50 (smoothed con peso 4 que ya tenían)
penalty_probs = {t: penalty_reported.get(t, 0.50) for t in team_names}
pd.DataFrame({
    "team": list(penalty_probs.keys()),
    "p_win_smoothed": list(penalty_probs.values()),
}).to_csv(WORKDIR / "penalty_probs.csv", index=False)

# ====================================================================
# draw_tendency.csv — del reporte v3
# ====================================================================
# Columnas que espera el script: team, diff, n
# diff = desviación empírica del empate (positivo = empata más, negativo = empata menos)
# n = tamaño de muestra (afecta el shrinkage en live_update)
draw_reported = {
    "Switzerland": (0.209, 80),
    "South Africa": (0.198, 60),
    "Uruguay": (0.102, 100),
    "Czech Republic": (0.086, 80),
    "Argentina": (-0.126, 120),
    # Finlandia no está en el Mundial, lo dejamos fuera
}
# Resto: diff=0, n=0 (el shrinkage anula su efecto: diff*n/(n+12) = 0)
draw_rows = []
for t in team_names:
    if t in draw_reported:
        diff, n_size = draw_reported[t]
    else:
        diff, n_size = 0.0, 0
    draw_rows.append({"team": t, "diff": diff, "n": n_size})
pd.DataFrame(draw_rows).to_csv(WORKDIR / "draw_tendency.csv", index=False)

# ====================================================================
# optimized_params.csv — del reporte v3
# ====================================================================
# - threshold_pd  = 0.38  (predecir empate si P(empate) >= este valor)
# - threshold_diff= 0.26  (Y |P(H)-P(A)| <= este valor)
# - draw_strength = 2.0   (factor de escala del ajuste de empate)
pd.DataFrame({
    "threshold_pd": [0.38],
    "threshold_diff": [0.26],
    "draw_strength": [2.0],
}).to_csv(WORKDIR / "optimized_params.csv", index=False)

print("\nArchivos generados:")
for f in ["dixon_coles_params.csv", "dixon_coles_global.csv", "penalty_probs.csv",
          "draw_tendency.csv", "optimized_params.csv", "teams_base.json", "matches_base.json"]:
    p = WORKDIR / f
    print(f"  {f:<30} {p.stat().st_size} bytes")
