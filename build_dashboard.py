"""
Regenera forescore_mundial_dashboard.html aplicando:
- Ajustes acumulados de live_state.json a las lambdas de partidos PENDIENTES
- Marca partidos JUGADOS con resultado real
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path

WORKDIR = Path(__file__).parent
SRC = WORKDIR / "dashboard_template.html"
OUT = WORKDIR / "forescore_mundial_dashboard.html"

dc = pd.read_csv(WORKDIR / "dixon_coles_params.csv", encoding="utf-8")
glob = pd.read_csv(WORKDIR / "dixon_coles_global.csv", encoding="utf-8")
alpha = dict(zip(dc["team"], dc["alpha"]))
beta = dict(zip(dc["team"], dc["beta"]))
RHO = float(glob["rho"].iloc[0])

state_file = WORKDIR / "live_state.json"
if state_file.exists():
    with open(state_file) as f: state = json.load(f)
else:
    state = {"alpha_adj":{}, "beta_adj":{}, "history":[]}

# Usar rho y mu recalibrados desde el torneo si están disponibles
RHO  = state.get("rho_live", RHO)
MU_H = state.get("mu_h", 1.0)
MU_A = state.get("mu_a", 1.0)

# Ventaja de anfitrion (consistente con live_update.compute_lambdas)
HOSTS_2026 = {"Mexico", "United States", "Canada"}
HOST_ADV = 1.25

alpha_adj = state.get("alpha_adj", {})
beta_adj = state.get("beta_adj", {})
history = state.get("history", [])
qualifiers = state.get("qualifiers", {})  # {slot: team_en}

with open(WORKDIR / "teams_base.json", encoding="utf-8") as f: teams_meta = json.load(f)
es_by_en = {t["team"]: t["team_es"] for t in teams_meta}
en_by_es = {t["team_es"]: t["team"] for t in teams_meta}

# Indexar jugados por (es,es) y (en,en)
played = {}
for h in history:
    h_en, a_en = h["home"], h["away"]
    h_es, a_es = es_by_en.get(h_en, h_en), es_by_en.get(a_en, a_en)
    played[(h_es, a_es)] = h
    played[(h_en, a_en)] = h

html = SRC.read_text(encoding="utf-8")
m_matches = re.search(r"const matches = (\[.*?\]);", html, re.DOTALL)
matches = json.loads(m_matches.group(1))

for mt in matches:
    h_es, a_es = mt["home_es"], mt["away_es"]
    h_en = en_by_es.get(h_es, h_es)
    a_en = en_by_es.get(a_es, a_es)
    key = (h_es, a_es)
    if key in played:
        rec = played[key]
        sh, sa = map(int, rec["score"].split("-"))
        mt["played"] = True
        mt["score_h"] = sh
        mt["score_a"] = sa
        mt["lambda_h_pred"] = rec["lh_pred"]
        mt["lambda_a_pred"] = rec["la_pred"]
        # Recalcular top-5 con los lambdas del momento del partido + corrección DC
        from math import exp, factorial
        def pois(l, k): return exp(-l) * (l**k) / factorial(min(k, 7))
        def dc_tau(h, a, lh, la, rho):
            if h==0 and a==0: return 1 - lh*la*rho
            if h==1 and a==0: return 1 + la*rho
            if h==0 and a==1: return 1 + lh*rho
            if h==1 and a==1: return 1 - rho
            return 1.0
        # Aplicar mu (igual que la prediccion pre-partido); lh_pred ya incluye host_adv.
        # Sin esto, el top-5 retrospectivo no coincide con el que se mostro antes del partido.
        lh, la = rec["lh_pred"] * MU_H, rec["la_pred"] * MU_A
        score_probs = {}
        for gh in range(8):
            for ga in range(8):
                score_probs[f"{gh}-{ga}"] = pois(lh, gh) * pois(la, ga) * dc_tau(gh, ga, lh, la, RHO)
        top5 = sorted(score_probs.items(), key=lambda x: -x[1])[:5]
        for i, (sc, pr) in enumerate(top5, 1):
            mt[f"score_{i}"] = sc
            mt[f"prob_{i}"]  = pr
    else:
        mt["played"] = False
        a_h = alpha.get(h_en, 0) + alpha_adj.get(h_en, 0)
        b_h = beta.get(h_en, 0) + beta_adj.get(h_en, 0)
        a_a = alpha.get(a_en, 0) + alpha_adj.get(a_en, 0)
        b_a = beta.get(a_en, 0) + beta_adj.get(a_en, 0)
        lh = float(np.exp(a_h - b_a)) * MU_H * (HOST_ADV if h_en in HOSTS_2026 else 1.0)
        la = float(np.exp(a_a - b_h)) * MU_A
        mt["lambda_h"] = lh
        mt["lambda_a"] = la
        # Recalcular top-5 marcadores con lambdas ajustados + corrección DC
        from math import exp, factorial
        def pois(l, k): return exp(-l) * (l**k) / factorial(min(k, 7))
        def dc_tau(h, a, lh, la, rho):
            if h==0 and a==0: return 1 - lh*la*rho
            if h==1 and a==0: return 1 + la*rho
            if h==0 and a==1: return 1 + lh*rho
            if h==1 and a==1: return 1 - rho
            return 1.0
        score_probs = {}
        for gh in range(8):
            for ga in range(8):
                score_probs[f"{gh}-{ga}"] = pois(lh, gh) * pois(la, ga) * dc_tau(gh, ga, lh, la, RHO)
        top5 = sorted(score_probs.items(), key=lambda x: -x[1])[:5]
        for i, (sc, pr) in enumerate(top5, 1):
            mt[f"score_{i}"] = sc
            mt[f"prob_{i}"]  = pr

# Reemplazar array matches
new_json = json.dumps(matches, ensure_ascii=False)
html_out = re.sub(r"const matches = \[.*?\];",
                  f"const matches = {new_json};",
                  html, count=1, flags=re.DOTALL)

# Inyectar clasificados confirmados (slot → nombre en español)
qualifiers_es = {slot: es_by_en.get(team, team) for slot, team in qualifiers.items()}
qual_json = json.dumps(qualifiers_es, ensure_ascii=False)
if re.search(r"const qualifiers = \{.*?\};", html_out, re.DOTALL):
    html_out = re.sub(r"const qualifiers = \{.*?\};",
                      f"const qualifiers = {qual_json};",
                      html_out, count=1, flags=re.DOTALL)
else:
    html_out = html_out.replace("const KO_R32 =",
                                f"const qualifiers = {qual_json};\nconst KO_R32 =", 1)

# El patch JS ya no es necesario — renderMatches() maneja los jugados directamente

# Banner si hay actividad
n_played = len(history)
n_adj = sum(1 for v in alpha_adj.values() if abs(v) > 0.005)
if n_played or n_adj:
    banner = f"""
<div style="background:#4a6b3e;color:#f5f1e8;padding:10px 32px;font-family:'JetBrains Mono',monospace;font-size:11px;text-transform:uppercase;letter-spacing:0.18em;text-align:center;">
LIVE · {n_played} partidos procesados · {n_adj} equipos ajustados
</div>"""
    html_out = html_out.replace("<body>", "<body>" + banner, 1)

OUT.write_text(html_out, encoding="utf-8")
print(f"Dashboard regenerado: {OUT.name}")
print(f"  Partidos jugados: {n_played}")
print(f"  Equipos con ajuste: {n_adj}")
