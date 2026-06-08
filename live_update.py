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
    dc = pd.read_csv(WORKDIR / "dixon_coles_params.csv")
    glob = pd.read_csv(WORKDIR / "dixon_coles_global.csv")
    opt = pd.read_csv(WORKDIR / "optimized_params.csv") if (WORKDIR/"optimized_params.csv").exists() else None
    pp = pd.read_csv(WORKDIR / "penalty_probs.csv")
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
        dt = pd.read_csv(dt_path)
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
    save_state(state)
    print(f"\nUPDATE: {home} {sh}-{sa} {away}")
    print(f"lambdas predichas: {lh_pred:.2f} - {la_pred:.2f}")
    print(f"\nAjustes Bayesianos aplicados (peso={BW}):")
    print(f"  alpha_{home}: {state['alpha_adj'][home]:+.3f}")
    print(f"  alpha_{away}: {state['alpha_adj'][away]:+.3f}")
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
    args = p.parse_args()
    if args.command == "predict": predict(args)
    elif args.command == "update": update(args)
    elif args.command == "status": status(args)
    elif args.command == "reset": reset_cmd(args)
    elif args.command == "rebuild-dashboard": rebuild(args)
    else: p.print_help()

if __name__ == "__main__":
    main()
