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

state_file = WORKDIR / "live_state.json"
if state_file.exists():
    with open(state_file) as f: state = json.load(f)
else:
    state = {"alpha_adj":{}, "beta_adj":{}, "history":[]}
alpha_adj = state.get("alpha_adj", {})
beta_adj = state.get("beta_adj", {})
history = state.get("history", [])

with open(WORKDIR / "teams_base.json") as f: teams_meta = json.load(f)
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
    else:
        mt["played"] = False
        a_h = alpha.get(h_en, 0) + alpha_adj.get(h_en, 0)
        b_h = beta.get(h_en, 0) + beta_adj.get(h_en, 0)
        a_a = alpha.get(a_en, 0) + alpha_adj.get(a_en, 0)
        b_a = beta.get(a_en, 0) + beta_adj.get(a_en, 0)
        mt["lambda_h"] = float(np.exp(a_h - b_a))
        mt["lambda_a"] = float(np.exp(a_a - b_h))

# Reemplazar array
new_json = json.dumps(matches, ensure_ascii=False)
html_out = re.sub(r"const matches = \[.*?\];",
                  f"const matches = {new_json};",
                  html, count=1, flags=re.DOTALL)

# Patch JS: marcar visualmente los partidos jugados
patch = """
// === live-update patch ===
(function(){
  function applyPlayed(){
    if (typeof matches === 'undefined') return;
    const playedMap = {};
    matches.forEach(m => { if(m.played) playedMap[m.home_es+'|'+m.away_es] = m; });
    document.querySelectorAll('tr, .match-row, .match-card').forEach(r => {
      const txt = r.textContent || '';
      Object.keys(playedMap).forEach(k => {
        const [h,a] = k.split('|');
        if (txt.includes(h) && txt.includes(a)) {
          const m = playedMap[k];
          r.style.background = 'rgba(74,107,62,0.10)';
          r.style.borderLeft = '3px solid #4a6b3e';
          // Inyectar marcador real si encontramos celdas de lambda
          const cells = r.querySelectorAll('td');
          cells.forEach(c => {
            if (c.textContent.match(/^\\d+\\.\\d+$/)) {
              // Reemplazar primera lambda por marcador
              if (!r.dataset.scored) {
                c.innerHTML = `<strong style="color:#4a6b3e">${m.score_h}–${m.score_a}</strong>`;
                r.dataset.scored = '1';
              }
            }
          });
        }
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(applyPlayed, 100));
  } else {
    setTimeout(applyPlayed, 100);
  }
  // Re-aplicar tras cualquier render dinámico
  const obs = new MutationObserver(() => setTimeout(applyPlayed, 50));
  obs.observe(document.body, {childList: true, subtree: true});
})();
"""
html_out = html_out.replace("</script>\n</body>", patch + "\n</script>\n</body>", 1)
if patch not in html_out:
    # fallback
    html_out = html_out.replace("</body>", f"<script>{patch}</script>\n</body>", 1)

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
