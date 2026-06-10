"""
apply_simulation.py — Inyecta los resultados de mc_final.csv en el array
'teams' del dashboard, recalculando p_advance/p_r16/.../p_champ.
Mantiene los campos de grupo, nombres y stats de grupo (avg_*).
Recalcula intervalos de confianza simples alrededor del nuevo p_champ.
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path

WORKDIR = Path(__file__).parent
SRC = WORKDIR / "dashboard_template.html"
OUT = WORKDIR / "forescore_mundial_dashboard.html"

mc = pd.read_csv(WORKDIR / "mc_final.csv").set_index("team")
pp = pd.read_csv(WORKDIR / "penalty_probs.csv").set_index("team")["p_win_smoothed"].to_dict()
html = SRC.read_text()
m = re.search(r"const teams = (\[.*?\]);", html, re.DOTALL)
teams = json.loads(m.group(1))

for t in teams:
    name = t["team"]
    t["pen_pct"] = float(pp.get(name, 0.50))
    if name in mc.index:
        r = mc.loc[name]
        t["p_1st"] = float(r["p_1st"]); t["p_2nd"] = float(r["p_2nd"]); t["p_3rd"] = float(r["p_3rd"])
        t["p_advance"] = float(r["p_advance"]); t["p_r16"] = float(r["p_r16"])
        t["p_qf"] = float(r["p_qf"]); t["p_sf"] = float(r["p_sf"])
        t["p_final"] = float(r["p_final"]); t["p_champ"] = float(r["p_champ"])
        # Intervalo aproximado: +- 1.96*sqrt(p(1-p)/n_eff). n_eff ~ equipo dependiente,
        # usamos banda heuristica proporcional para mantener el campo visual.
        p = float(r["p_champ"])
        band = 1.96 * np.sqrt(max(p*(1-p),1e-6)/300)  # banda visual
        t["p_champ_mean"] = p
        t["p_champ_lo"] = max(0, p - band)
        t["p_champ_hi"] = min(1, p + band)

new_json = json.dumps(teams, ensure_ascii=False)
html_out = re.sub(r"const teams = \[.*?\];", f"const teams = {new_json};",
                  html, count=1, flags=re.DOTALL)

# Nota de versión: marcar que la simulación usa bracket oficial + mejoras
html_out = html_out.replace(
    "50.000 SIMS · PENALES + EMPATE CALIBRADOS",
    "50.000 SIMS · BRACKET OFICIAL · VARIANZA + SHRINKAGE"
)

OUT.write_text(html_out)
print(f"Dashboard actualizado con simulación nueva: {OUT.name}")
top = sorted(teams, key=lambda x:-x["p_champ"])[:5]
for t in top:
    print(f"  {t['team_es']:<14} {t['p_champ']*100:.1f}%")
