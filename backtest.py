"""
backtest.py - Forescore Mundial 2026
====================================

Evalua el modelo sobre los partidos ya jugados (live_state.json -> history)
replicando EXACTAMENTE la regla de prediccion de live_update.predict:

    lambdas base (alpha/beta del momento, ya guardadas en lh_pred/la_pred)
      * mu_h/mu_a (inflacion de goles aprendida)
      -> match_prob con rho
      -> ajuste de empate por equipo (draw_tendency.csv)
      -> umbral de empate (threshold_pd / threshold_diff)

Modos
-----
  walk-forward (por defecto): para cada partido i, mu y rho se recalibran
      usando SOLO los partidos 0..i-1. Sin fuga de datos (honesto).
  --in-sample: usa el mu/rho final del estado para todos los partidos
      (mas rapido, pero optimista: el partido se evalua con info que lo incluye).

Metricas
--------
  Acierto 1X2     : % de veces que la prediccion (H/D/A) coincide con el real.
  Marcador exacto : % con el marcador mas probable == real.
  MAE / RMSE goles: error medio por equipo en goles.
  Log-loss 1X2    : -mean(log p(resultado real)). Menor = mejor. Premia calibracion.
  Brier 1X2       : mean sum_k (p_k - y_k)^2 en {0,2}. Menor = mejor.
  Goles pred/real : media de goles por partido, para ver sesgo de calibracion.

Uso
---
  python backtest.py                 # walk-forward, modelo actual
  python backtest.py --in-sample     # rapido, con fuga (diagnostico)
  python backtest.py --no-mu         # desactiva la inflacion de goles
  python backtest.py --rho -0.092    # fuerza un rho fijo (ignora el aprendido)
  python backtest.py --no-draw-adj   # sin ajuste de empate por equipo
"""
import argparse
import contextlib
import io
import math

import numpy as np
import pandas as pd

import live_update as L

WORKDIR = L.WORKDIR


def load_draw_adj():
    p = WORKDIR / "draw_tendency.csv"
    if not p.exists():
        return {}
    dt = pd.read_csv(p, encoding="utf-8")
    SHRINK = 12
    dt["adj"] = dt["diff"] * (dt["n"] / (dt["n"] + SHRINK))
    return dict(zip(dt["team"], dt["adj"]))


def outcome(gh, ga):
    return "H" if gh > ga else ("A" if gh < ga else "D")


def predict_triplet(lh, la, home, away, base, rho, draw_adj, use_draw_adj):
    """Devuelve (pred, (pH, pD, pA)) replicando la regla de live_update.predict."""
    ph, pd_, pa, pm = L.match_prob(lh, la, rho)
    if use_draw_adj and draw_adj:
        combined = (draw_adj.get(home, 0) + draw_adj.get(away, 0)) / 2
        pd_adj = float(np.clip(pd_ + combined * base["draw_strength"], 0.05, 0.85))
        nd = ph + pa
        scale = (1 - pd_adj) / nd if nd > 0 else 1
        ph_adj, pa_adj = ph * scale, pa * scale
    else:
        ph_adj, pd_adj, pa_adj = ph, pd_, pa
    # Regla de decision identica a predict()
    if pd_adj >= base["threshold_pd"] and abs(ph_adj - pa_adj) <= base["threshold_diff"]:
        pred = "D"
    elif ph_adj > pa_adj:
        pred = "H"
    else:
        pred = "A"
    # Marcador mas probable
    flat = [(h, a, pm[h, a]) for h in range(pm.shape[0]) for a in range(pm.shape[1])]
    best = max(flat, key=lambda x: x[2])
    return pred, (ph_adj, pd_adj, pa_adj), (best[0], best[1])


def calib_for_prefix(history_prefix):
    """mu_h, mu_a, rho_live como los habria visto el modelo con solo estos partidos."""
    tmp = {"history": history_prefix}
    with contextlib.redirect_stdout(io.StringIO()):
        L.recalibrate_params(tmp)
    return tmp.get("mu_h", 1.0), tmp.get("mu_a", 1.0), tmp.get("rho_live", -0.092)


def run(args):
    base = L.load_base()
    state = L.load_state()
    history = state.get("history", [])
    if not history:
        print("No hay partidos en el historial.")
        return
    draw_adj = load_draw_adj()
    use_draw_adj = not args.no_draw_adj

    # Calibracion para modo in-sample (final)
    mu_h_f = 1.0 if args.no_mu else state.get("mu_h", 1.0)
    mu_a_f = 1.0 if args.no_mu else state.get("mu_a", 1.0)
    rho_f = args.rho if args.rho is not None else state.get("rho_live", base["rho"])

    n = len(history)
    hits = exact = 0
    se_goals = ae_goals = 0.0
    logloss = brier = 0.0
    g_pred = g_real = 0.0
    rows = []

    for i, m in enumerate(history):
        gh, ga = map(int, m["score"].split("-"))
        home = L.resolve_team(m["home"], base)
        away = L.resolve_team(m["away"], base)
        re = outcome(gh, ga)

        if args.mode == "walk" and not args.no_mu:
            mu_h, mu_a, rho = calib_for_prefix(history[:i])
            if args.rho is not None:
                rho = args.rho
        elif args.mode == "walk":  # no_mu pero walk: solo rho walk-forward
            _, _, rho = calib_for_prefix(history[:i])
            mu_h = mu_a = 1.0
            if args.rho is not None:
                rho = args.rho
        else:  # in-sample
            mu_h, mu_a, rho = mu_h_f, mu_a_f, rho_f

        # lambdas base de pre-partido (ya guardadas) * inflacion de goles * ventaja anfitrion
        lh = m["lh_pred"] * mu_h
        la = m["la_pred"] * mu_a
        if not args.no_host and home in L.HOSTS_2026:
            lh *= L.HOST_ADV

        pred, (pH, pD, pA), (bh, ba) = predict_triplet(
            lh, la, home, away, base, rho, draw_adj, use_draw_adj)

        hits += pred == re
        exact += (bh == gh and ba == ga)
        ae_goals += abs(lh - gh) + abs(la - ga)
        se_goals += (lh - gh) ** 2 + (la - ga) ** 2
        g_pred += lh + la
        g_real += gh + ga

        probs = {"H": pH, "D": pD, "A": pA}
        p_true = max(probs[re], 1e-12)
        logloss += -math.log(p_true)
        brier += sum((probs[k] - (1.0 if k == re else 0.0)) ** 2 for k in "HDA")

        rows.append((("OK " if pred == re else "XX "), pred, re,
                     gh, ga, pH, pD, pA, m["home"], m["away"]))

    print(f"\n{'='*70}")
    print(f"  BACKTEST  ({'walk-forward' if args.mode=='walk' else 'in-sample'}"
          f"{', sin mu' if args.no_mu else ''}"
          f"{f', rho fijo={args.rho}' if args.rho is not None else ''}"
          f"{', sin ajuste empate' if args.no_draw_adj else ''})")
    print(f"{'='*70}")
    if args.verbose:
        print(f"\n{'':3} pr re  marc   P(H)  P(D)  P(A)   partido")
        for r in rows:
            print(f"{r[0]}{r[1]}->{r[2]}  {r[3]}-{r[4]}  "
                  f"{r[5]*100:4.0f}% {r[6]*100:4.0f}% {r[7]*100:4.0f}%  {r[8]} vs {r[9]}")
    print(f"\n  Partidos evaluados : {n}")
    print(f"  Acierto 1X2        : {hits}/{n} = {100*hits/n:.1f}%")
    print(f"  Marcador exacto    : {exact}/{n} = {100*exact/n:.1f}%")
    print(f"  MAE goles/equipo   : {ae_goals/(2*n):.3f}")
    print(f"  RMSE goles/equipo  : {math.sqrt(se_goals/(2*n)):.3f}")
    print(f"  Log-loss 1X2       : {logloss/n:.4f}   (menor = mejor; azar 3-vias = 1.0986)")
    print(f"  Brier 1X2          : {brier/n:.4f}   (menor = mejor; azar 3-vias = 0.6667)")
    print(f"  Goles/partido pred : {g_pred/n:.2f}   real: {g_real/n:.2f}   "
          f"sesgo: {(g_pred-g_real)/n:+.2f}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Backtest del modelo Forescore Mundial 2026")
    ap.add_argument("--in-sample", dest="mode", action="store_const",
                    const="insample", default="walk",
                    help="usa mu/rho final (rapido, optimista). Por defecto: walk-forward.")
    ap.add_argument("--no-mu", action="store_true", help="desactiva la inflacion de goles")
    ap.add_argument("--no-draw-adj", action="store_true",
                    help="desactiva el ajuste de empate por equipo")
    ap.add_argument("--no-host", action="store_true",
                    help="desactiva la ventaja de anfitrion")
    ap.add_argument("--rho", type=float, default=None,
                    help="fuerza un rho fijo (p.ej. -0.092 para el prior)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="muestra el detalle partido a partido")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
