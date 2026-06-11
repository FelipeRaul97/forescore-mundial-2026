"""
Forescore-Mundial v3.0 — Pipeline de actualización en vivo
(versión adaptada: acepta nombres es/en, regenera dashboard tras cada update)
"""
import json, argparse, sys, subprocess
from datetime import datetime
import pandas as pd, numpy as np
from scipy.stats import poisson
from pathlib import Path

WORKDIR = Path(__file__).parent
STATE_FILE = WORKDIR / "live_state.json"

def load_base():
    dc = pd.read_csv(WORKDIR / "dixon_coles_params.csv", encoding="utf-8")
    glob = pd.read_csv(WORKDIR / "dixon_coles_global.csv", encoding="utf-8")
    opt = pd.read_csv(WORKDIR / "optimized_params.csv", encoding="utf-8") if (WORKDIR/"optimized_params.csv").exists() else None
    pp = pd.read_csv(WORKDIR / "penalty_probs.csv", encoding="utf-8")
    with open(WORKDIR / "teams_base.json") as f: teams_meta = json.load(f)
    es_by_en = {t["team"]: t["team_es"] for t in teams_meta}
    en_by_es = {t["team_es"]: t["team"] for t in teams_meta}
    return {
        "teams": dc["team"].tolist(),
        "alpha": dict(zip(dc["team"], dc["alpha"])),
        "beta": dict(zip(dc["team"], dc["beta"])),
        "gamma": float(glob["gamma"].iloc[0]),
        "rho": float(glob["rho"].iloc[0]),
        "draw_strength": float(opt["draw_strength"].iloc[0]) if opt is not None else 2.0,
        "threshold_pd": float(opt["threshold_pd"].iloc[0]) if opt is not None else 0.38,
        "threshold_diff": float(opt["threshold_diff"].iloc[0]) if opt is not None else 0.26,
        "penalty_probs": dict(zip(pp["team"], pp["p_win_smoothed"])),
        "es_by_en": es_by_en,
        "en_by_es": en_by_es,
    }

def resolve_team(name, base):
    """Acepta nombre en español o inglés y devuelve el canónico (inglés)."""
    if name in base["alpha"]: return name
    if name in base["en_by_es"]: return base["en_by_es"][name]
    lo = name.lower()
    for t in base["alpha"]:
        if t.lower() == lo: return t
    for es, en in base["en_by_es"].items():
        if es.lower() == lo: return en
    return None

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f: return json.load(f)
    base = load_base()
    state = {"alpha_adj": {k:0.0 for k in base["teams"]},
             "beta_adj": {k:0.0 for k in base["teams"]},
             "history": [], "last_updated": datetime.now().isoformat()}
    save_state(state)
    return state

def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE,"w") as f: json.dump(state, f, indent=2, ensure_ascii=False)

def compute_lambdas(home, away, base, state, **kw):
    if home not in base["alpha"] or away not in base["alpha"]:
        return None, None, "Equipo no en modelo"
    a_h = base["alpha"][home] + state["alpha_adj"].get(home, 0)
    b_h = base["beta"][home] + state["beta_adj"].get(home, 0)
    a_a = base["alpha"][away] + state["alpha_adj"].get(away, 0)
    b_a = base["beta"][away] + state["beta_adj"].get(away, 0)
    gf = base["gamma"] if kw.get("is_home_country") else 0
    log_lh = a_h - b_a + gf
    log_la = a_a - b_h
    adj = []
    if kw.get("absences_home"):
        items = kw["absences_home"].split(",") if isinstance(kw["absences_home"], str) else kw["absences_home"]
        n = len([x for x in items if x.strip()])
        if n > 0:
            factor = max(0.5, 1-0.10*n)
            log_lh += np.log(factor)
            adj.append(f"Home ausencias ({n}): lambda_h *{factor:.2f}")
    if kw.get("absences_away"):
        items = kw["absences_away"].split(",") if isinstance(kw["absences_away"], str) else kw["absences_away"]
        n = len([x for x in items if x.strip()])
        if n > 0:
            factor = max(0.5, 1-0.10*n)
            log_la += np.log(factor)
            adj.append(f"Away ausencias ({n}): lambda_a *{factor:.2f}")
    if kw.get("backup_gk_home"):
        log_la += np.log(1.08); adj.append("Backup GK home: lambda_a *1.08")
    if kw.get("backup_gk_away"):
        log_lh += np.log(1.08); adj.append("Backup GK away: lambda_h *1.08")
    if kw.get("rotation_home"):
        log_lh += np.log(0.95); adj.append("Rotacion home: lambda_h *0.95")
    if kw.get("rotation_away"):
        log_la += np.log(0.95); adj.append("Rotacion away: lambda_a *0.95")
    if kw.get("weather_extreme"):
        log_lh += np.log(0.95); log_la += np.log(0.95)
        adj.append("Clima extremo: ambos *0.95")
    return float(np.exp(log_lh)), float(np.exp(log_la)), adj

def match_prob(lh, la, rho):
    p = np.zeros((9,9))
    for h in range(9):
        for a in range(9):
            p[h,a] = poisson.pmf(h,lh)*poisson.pmf(a,la)
            if h==0 and a==0: p[h,a] *= (1-lh*la*rho)
            elif h==1 and a==0: p[h,a] *= (1+la*rho)
            elif h==0 and a==1: p[h,a] *= (1+lh*rho)
            elif h==1 and a==1: p[h,a] *= (1-rho)
    p = np.maximum(p, 0); p /= p.sum()
    return np.tril(p,-1).sum(), np.diag(p).sum(), np.triu(p,1).sum(), p

def predict(args):
    base = load_base(); state = load_state()
    home = resolve_team(args.home, base)
    away = resolve_team(args.away, base)
    if not home or not away:
        print(f"ERROR: equipo no encontrado (home={args.home}, away={args.away})")
        print(f"Disponibles (es): {', '.join(sorted(base['en_by_es'].keys())[:10])}...")
        return
    lh, la, adj = compute_lambdas(home, away, base, state,
        absences_home=args.absences_home, absences_away=args.absences_away,
        backup_gk_home=args.backup_gk_home, backup_gk_away=args.backup_gk_away,
        rotation_home=args.rotation_home, rotation_away=args.rotation_away,
        weather_extreme=args.weather_extreme, is_home_country=args.home_country)
    ph, pd_, pa, pm = match_prob(lh, la, base["rho"])
    # Ajuste de empate por equipo (v3)
    dt_path = WORKDIR / "draw_tendency.csv"
    if dt_path.exists():
        dt = pd.read_csv(dt_path, encoding="utf-8")
        SHRINK = 12
        dt["adj"] = dt["diff"] * (dt["n"]/(dt["n"]+SHRINK))
        da = dict(zip(dt["team"], dt["adj"]))
        combined = (da.get(home,0)+da.get(away,0))/2
        pd_adj = float(np.clip(pd_ + combined*base["draw_strength"], 0.05, 0.85))
        nd = ph + pa
        scale = (1-pd_adj)/nd if nd > 0 else 1
        ph_adj = ph * scale; pa_adj = pa * scale
    else:
        ph_adj, pd_adj, pa_adj = ph, pd_, pa

    flat = [(h,a,pm[h,a]) for h in range(9) for a in range(9)]
    flat.sort(key=lambda x:-x[2]); top5 = flat[:5]
    home_es = base["es_by_en"][home]; away_es = base["es_by_en"][away]

    print(f"\n{'='*60}")
    print(f"  PREDICCION: {home_es} vs {away_es}")
    print(f"{'='*60}")
    print(f"\nlambdas: {home_es}={lh:.2f} | {away_es}={la:.2f}")
    print(f"Goles totales esperados: {lh+la:.2f}")
    if adj:
        print("\nAjustes aplicados:")
        for a in adj: print(f"  - {a}")
    print(f"\nProbabilidades 1X2 (con ajuste empate v3):")
    print(f"  Victoria {home_es}: {ph_adj*100:.1f}%")
    print(f"  Empate: {pd_adj*100:.1f}%")
    print(f"  Victoria {away_es}: {pa_adj*100:.1f}%")
    if pd_adj >= base["threshold_pd"] and abs(ph_adj-pa_adj) <= base["threshold_diff"]:
        prediction = "EMPATE"
    elif ph_adj > pa_adj:
        prediction = f"GANA {home_es}"
    else:
        prediction = f"GANA {away_es}"
    print(f"\n>>> PREDICCION: {prediction} <<<")
    # Tanda de penales (info, solo eliminatorias)
    if args.knockout:
        p_pen_home = base["penalty_probs"].get(home, 0.50)
        print(f"\nSi llega a penales: P({home_es}) = {p_pen_home*100:.0f}% | P({away_es}) = {(1-p_pen_home)*100:.0f}%")
        # Nota: usa solo home como referencia; combinarlas correctamente requiere normalización
    print(f"\nTop-5 marcadores:")
    for h, a, p in top5: print(f"  {h}-{a}: {p*100:.1f}%")
    print(f"  (top-5 cubre {sum(p for _,_,p in top5)*100:.0f}%)")

def update(args):
    base = load_base(); state = load_state()
    home = resolve_team(args.home, base)
    away = resolve_team(args.away, base)
    if not home or not away:
        print(f"ERROR: equipo no encontrado"); return
    sh, sa = map(int, args.score.split("-"))
    a_h = base["alpha"][home] + state["alpha_adj"].get(home, 0)
    b_h = base["beta"][home] + state["beta_adj"].get(home, 0)
    a_a = base["alpha"][away] + state["alpha_adj"].get(away, 0)
    b_a = base["beta"][away] + state["beta_adj"].get(away, 0)
    gf = base["gamma"] if args.home_country else 0
    lh_pred = np.exp(a_h - b_a + gf)
    la_pred = np.exp(a_a - b_h)
    BW = 0.30
    diff_h = np.log(max(sh, 0.3)) - np.log(lh_pred)
    diff_a = np.log(max(sa, 0.3)) - np.log(la_pred)
    state["alpha_adj"][home] = state["alpha_adj"].get(home, 0) + BW*diff_h/2
    state["beta_adj"][away]  = state["beta_adj"].get(away, 0)  - BW*diff_h/2
    state["alpha_adj"][away] = state["alpha_adj"].get(away, 0) + BW*diff_a/2
    state["beta_adj"][home]  = state["beta_adj"].get(home, 0)  - BW*diff_a/2
    state["history"].append({"date": datetime.now().isoformat(), "home": home, "away": away,
                             "score": args.score, "lh_pred": float(lh_pred), "la_pred": float(la_pred)})

    # Detectar grupos completados y registrar clasificados W: y R:
    newly_qualified = auto_qualify(state, base)

    save_state(state)
    print(f"\nUPDATE: {home} {sh}-{sa} {away}")
    print(f"lambdas predichas: {lh_pred:.2f} - {la_pred:.2f}")
    print(f"\nAjustes Bayesianos aplicados (peso={BW}):")
    print(f"  alpha_{home}: {state['alpha_adj'][home]:+.3f}")
    print(f"  alpha_{away}: {state['alpha_adj'][away]:+.3f}")
    if newly_qualified:
        print(f"\nGrupos cerrados — clasificados registrados automáticamente:")
        for group, w, r in newly_qualified:
            print(f"  Grupo {group}: 1º {w}  ·  2º {r}")
        print("  (Los mejores terceros debes registrarlos con: partido.bat qualify \"Equipo\" \"T:XXXX\")")
    # Regenerar dashboard
    print("\nRegenerando dashboard...")
    try:
        subprocess.run([sys.executable, str(WORKDIR/"build_dashboard.py")], check=True)
    except Exception as e:
        print(f"WARN: no se pudo regenerar dashboard: {e}")

def status(args):
    state = load_state(); base = load_base()
    print(f"\nUltima actualizacion: {state.get('last_updated', 'nunca')}")
    print(f"Partidos procesados: {len(state['history'])}")
    adjs = [(t, state["alpha_adj"].get(t,0), state["beta_adj"].get(t,0)) for t in base["teams"]]
    adjs = [(t,a,b) for t,a,b in adjs if abs(a)+abs(b) > 0.005]
    adjs.sort(key=lambda x:-(abs(x[1])+abs(x[2])))
    if adjs:
        print("\nEquipos con mayor ajuste:")
        for t,a,b in adjs[:15]:
            print(f"  {t:<22} alpha={a:+.3f} beta={b:+.3f}")
    else:
        print("\nSin ajustes acumulados todavia.")
    if state["history"]:
        print(f"\nUltimos 5 partidos:")
        for h in state["history"][-5:]:
            print(f"  {h['date'][:10]}: {h['home']} {h['score']} {h['away']}")

def reset_cmd(args):
    if STATE_FILE.exists():
        STATE_FILE.unlink(); print("Estado reseteado.")
    else:
        print("No habia estado previo.")
    print("Regenerando dashboard base...")
    try:
        subprocess.run([sys.executable, str(WORKDIR/"build_dashboard.py")], check=True)
    except Exception as e:
        print(f"WARN: {e}")

def rebuild(args):
    subprocess.run([sys.executable, str(WORKDIR/"build_dashboard.py")], check=True)

def resimulate(args):
    """Re-corre el Monte Carlo completo con los alpha/beta actuales
    (base + ajustes acumulados + anclaje Elo) y reescribe las P(campeon)/fases."""
    print(f"Re-anclando a Elo (w={args.elo_w})...")
    subprocess.run([sys.executable, str(WORKDIR/"elo_anchor.py"), "--w", str(args.elo_w)],
                   check=True)
    print(f"Re-simulando torneo ({args.sims} sims, k={args.k}, shrink={args.shrink})...")
    subprocess.run([sys.executable, str(WORKDIR/"monte_carlo.py"),
                    "--sims", str(args.sims), "--k", str(args.k),
                    "--shrink", str(args.shrink), "--out", str(WORKDIR/"mc_final.csv")],
                   check=True)
    print("Regenerando dashboard con resultados reales...")
    subprocess.run([sys.executable, str(WORKDIR/"build_dashboard.py")], check=True)
    print("Aplicando simulación al dashboard...")
    subprocess.run([sys.executable, str(WORKDIR/"apply_simulation.py")], check=True)
    print("Listo. Recuerda: git add forescore_mundial_dashboard.html && git commit && git push")

def auto_qualify(state, base):
    """Detecta grupos completados y registra W: y R: automáticamente."""
    import re, json
    from pathlib import Path

    # Cargar estructura de partidos y equipos del template
    template = WORKDIR / "dashboard_template.html"
    html = template.read_text(encoding="utf-8")
    m = re.search(r"const matches = (\[.*?\]);", html, re.DOTALL)
    m2 = re.search(r"const teams = (\[.*?\]);", html, re.DOTALL)
    if not m or not m2:
        return []

    all_matches = json.loads(m.group(1))
    all_teams = json.loads(m2.group(1))

    # Índice equipo_es → grupo e inglés
    es_to_group = {t["team_es"]: t["group"] for t in all_teams}
    es_to_en = {t["team_es"]: t["team"] for t in all_teams}

    # Partidos jugados según historial
    played_pairs = set()
    for h in state["history"]:
        h_es = base["es_by_en"].get(h["home"], h["home"])
        a_es = base["es_by_en"].get(h["away"], h["away"])
        played_pairs.add((h_es, a_es))

    # Agrupar partidos del template por grupo
    from collections import defaultdict
    group_matches = defaultdict(list)
    for mt in all_matches:
        g = es_to_group.get(mt["home_es"])
        if g:
            group_matches[g].append(mt)

    if "qualifiers" not in state:
        state["qualifiers"] = {}

    newly_qualified = []
    for group, gmatches in group_matches.items():
        # Grupo completo = los 6 partidos jugados
        if len(gmatches) != 6:
            continue
        all_played = all(
            (mt["home_es"], mt["away_es"]) in played_pairs or
            (mt["away_es"], mt["home_es"]) in played_pairs
            for mt in gmatches
        )
        if not all_played:
            continue

        # Ya están registrados?
        if f"W:{group}" in state["qualifiers"] and f"R:{group}" in state["qualifiers"]:
            continue

        # Calcular tabla real
        points = defaultdict(int)
        gd = defaultdict(int)
        gf = defaultdict(int)
        for mt in gmatches:
            h_es, a_es = mt["home_es"], mt["away_es"]
            # Buscar score en historial
            score = None
            for h in state["history"]:
                h_es2 = base["es_by_en"].get(h["home"], h["home"])
                a_es2 = base["es_by_en"].get(h["away"], h["away"])
                if (h_es2, a_es2) == (h_es, a_es) or (a_es2, h_es2) == (h_es, a_es):
                    score = h["score"]
                    if (a_es2, h_es2) == (h_es, a_es):
                        parts = score.split("-")
                        score = f"{parts[1]}-{parts[0]}"
                    break
            if not score:
                continue
            sh, sa = map(int, score.split("-"))
            if sh > sa:
                points[h_es] += 3
            elif sh == sa:
                points[h_es] += 1; points[a_es] += 1
            else:
                points[a_es] += 3
            gd[h_es] += sh - sa; gd[a_es] += sa - sh
            gf[h_es] += sh; gf[a_es] += sa

        group_team_names = [mt["home_es"] for mt in gmatches[:4:2]] + \
                           [mt["away_es"] for mt in gmatches[:4:2]]
        group_team_names = list({es_to_group.get(t) and t for t in
                                  [mt["home_es"] for mt in gmatches] +
                                  [mt["away_es"] for mt in gmatches]
                                  if es_to_group.get(t) == group})

        standings = sorted(
            group_team_names,
            key=lambda t: (points[t], gd[t], gf[t]),
            reverse=True
        )
        if len(standings) < 2:
            continue

        winner_es, runner_es = standings[0], standings[1]
        winner_en = es_to_en.get(winner_es, winner_es)
        runner_en = es_to_en.get(runner_es, runner_es)

        state["qualifiers"][f"W:{group}"] = winner_en
        state["qualifiers"][f"R:{group}"] = runner_en
        newly_qualified.append((group, winner_es, runner_es))

    return newly_qualified


def qualify(args):
    """Registra un clasificado confirmado a la fase eliminatoria."""
    base = load_base()
    team_en = resolve_team(args.team, base)
    if not team_en:
        print(f"ERROR: equipo '{args.team}' no reconocido."); return
    slot = args.slot.upper()  # e.g. W:A, R:B, T:ABCDF
    valid_types = ("W:", "R:", "T:")
    if not any(slot.startswith(t) for t in valid_types):
        print(f"ERROR: slot debe ser W:X, R:X o T:XXXX (ej: W:A, R:B, T:ABCDF)"); return
    state = load_state()
    if "qualifiers" not in state:
        state["qualifiers"] = {}
    state["qualifiers"][slot] = team_en
    save_state(state)
    team_es = base["es_by_en"].get(team_en, team_en)
    print(f"Clasificado registrado: {team_es} → slot {slot}")
    print("Regenerando dashboard...")
    try:
        subprocess.run([sys.executable, str(WORKDIR/"build_dashboard.py")], check=True)
    except Exception as e:
        print(f"WARN: {e}")

def main():
    p = argparse.ArgumentParser(description="Forescore v3.0 - live update")
    sub = p.add_subparsers(dest="command")
    pp = sub.add_parser("predict")
    pp.add_argument("--home", required=True)
    pp.add_argument("--away", required=True)
    pp.add_argument("--absences-home", default=None)
    pp.add_argument("--absences-away", default=None)
    pp.add_argument("--backup-gk-home", action="store_true")
    pp.add_argument("--backup-gk-away", action="store_true")
    pp.add_argument("--rotation-home", action="store_true")
    pp.add_argument("--rotation-away", action="store_true")
    pp.add_argument("--weather-extreme", action="store_true")
    pp.add_argument("--home-country", action="store_true")
    pp.add_argument("--knockout", action="store_true", help="Mostrar penalty prob si es eliminatoria")
    pu = sub.add_parser("update")
    pu.add_argument("--home", required=True)
    pu.add_argument("--away", required=True)
    pu.add_argument("--score", required=True)
    pu.add_argument("--home-country", action="store_true")
    sub.add_parser("status")
    sub.add_parser("reset")
    sub.add_parser("rebuild-dashboard")
    pq = sub.add_parser("qualify", help="Registrar clasificado confirmado a eliminatorias")
    pq.add_argument("--team", required=True, help="Nombre del equipo")
    pq.add_argument("--slot", required=True, help="Slot del bracket: W:A (1ro grupo A), R:B (2do grupo B), T:ABCDF (mejor tercero)")
    prs = sub.add_parser("resimulate", help="Re-correr Monte Carlo y actualizar P(campeon)")
    prs.add_argument("--sims", type=int, default=50000)
    prs.add_argument("--k", type=float, default=18.0)
    prs.add_argument("--shrink", type=float, default=0.15)
    prs.add_argument("--elo-w", type=float, default=0.25)
    args = p.parse_args()
    if args.command == "predict": predict(args)
    elif args.command == "update": update(args)
    elif args.command == "status": status(args)
    elif args.command == "reset": reset_cmd(args)
    elif args.command == "rebuild-dashboard": rebuild(args)
    elif args.command == "qualify": qualify(args)
    elif args.command == "resimulate": resimulate(args)
    else: p.print_help()

if __name__ == "__main__":
    main()
