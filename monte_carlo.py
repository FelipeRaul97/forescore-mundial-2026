"""
monte_carlo.py — Motor de simulacion del Mundial 2026 con BRACKET OFICIAL.

Mejoras incorporadas (activables):
  1) Varianza inflada (Gamma-Poisson, parametro k)
  2) Shrinkage de alpha/beta hacia media de confederacion

Usa el bracket fijo oficial (bracket_2026.py). Los 8 mejores terceros se
asignan a sus slots respetando los clusters de grupo permitidos.
"""
import json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
import bracket_2026 as BR

WORKDIR = Path(__file__).parent

CONFED = {
    "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Uruguay":"CONMEBOL","Colombia":"CONMEBOL",
    "Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
    "Spain":"UEFA","France":"UEFA","England":"UEFA","Germany":"UEFA","Portugal":"UEFA",
    "Netherlands":"UEFA","Croatia":"UEFA","Belgium":"UEFA","Switzerland":"UEFA",
    "Austria":"UEFA","Norway":"UEFA","Sweden":"UEFA","Czech Republic":"UEFA",
    "Scotland":"UEFA","Turkey":"UEFA","Bosnia and Herzegovina":"UEFA",
    "Mexico":"CONCACAF","United States":"CONCACAF","Canada":"CONCACAF","Panama":"CONCACAF",
    "Haiti":"CONCACAF","Curacao":"CONCACAF",
    "Morocco":"CAF","Senegal":"CAF","Egypt":"CAF","Ivory Coast":"CAF","Algeria":"CAF",
    "Tunisia":"CAF","South Africa":"CAF","Ghana":"CAF","DR Congo":"CAF","Cape Verde":"CAF",
    "Japan":"AFC","South Korea":"AFC","Iran":"AFC","Saudi Arabia":"AFC","Australia":"AFC",
    "Qatar":"AFC","Uzbekistan":"AFC","Iraq":"AFC","Jordan":"AFC",
    "New Zealand":"OFC",
}

def load_params():
    # Preferir parametros ancladas a Elo si existen
    pf = WORKDIR/"dixon_coles_params_elo.csv"
    if not pf.exists():
        pf = WORKDIR/"dixon_coles_params.csv"
    dc = pd.read_csv(pf, encoding="utf-8")
    glob = pd.read_csv(WORKDIR/"dixon_coles_global.csv", encoding="utf-8")
    # Normalizar nombres (evita corrupcion de caracteres especiales en Windows)
    dc["team"] = dc["team"].str.encode("utf-8", errors="replace").str.decode("utf-8")
    alpha = dict(zip(dc["team"], dc["alpha"]))
    beta = dict(zip(dc["team"], dc["beta"]))
    rho = float(glob["rho"].iloc[0])
    mu_h, mu_a = 1.0, 1.0
    played = {}  # {(home_en, away_en): (gh, ga)}
    sf = WORKDIR/"live_state.json"
    if sf.exists():
        st = json.load(open(sf))
        for t,v in st.get("alpha_adj",{}).items(): alpha[t] = alpha.get(t,0)+v
        for t,v in st.get("beta_adj",{}).items():  beta[t]  = beta.get(t,0)+v
        if "rho_live" in st: rho = st["rho_live"]
        mu_h = st.get("mu_h", 1.0)
        mu_a = st.get("mu_a", 1.0)
        for h in st.get("history", []):
            gh, ga = map(int, h["score"].split("-"))
            played[(h["home"], h["away"])] = (gh, ga)
    return alpha, beta, rho, mu_h, mu_a, played

def apply_shrinkage(alpha, beta, lam=0.30):
    confs = {}
    for t in alpha:
        confs.setdefault(CONFED.get(t,"OTHER"), []).append(t)
    a_sh, b_sh = dict(alpha), dict(beta)
    for c, members in confs.items():
        if len(members) < 2: continue
        ma = np.mean([alpha[t] for t in members])
        mb = np.mean([beta[t] for t in members])
        for t in members:
            a_sh[t] = (1-lam)*alpha[t] + lam*ma
            b_sh[t] = (1-lam)*beta[t] + lam*mb
    return a_sh, b_sh

def simulate(n_sims=50000, k_disp=8.0, shrink=0.30, seed=42):
    alpha, beta, rho, mu_h, mu_a, played_scores = load_params()
    if shrink > 0:
        alpha, beta = apply_shrinkage(alpha, beta, lam=shrink)
    teams_meta = json.load(open(WORKDIR/"teams_base.json"))
    groups = {}
    for t in teams_meta:
        groups.setdefault(t["group"], []).append(t["team"])
    all_teams = [t["team"] for t in teams_meta]
    rng = np.random.default_rng(seed)
    pp = pd.read_csv(WORKDIR/"penalty_probs.csv", encoding="utf-8")
    ppd = dict(zip(pp["team"], pp["p_win_smoothed"]))

    cnt = {t: {"adv":0,"r16":0,"qf":0,"sf":0,"final":0,"champ":0,"1st":0,"2nd":0,"3rd":0}
           for t in all_teams}
    group_list = sorted(groups.keys())
    gfix = [(i,j) for i in range(4) for j in range(i+1,4)]

    def play(h, a):
        lh = np.exp(alpha[h]-beta[a]) * mu_h; la = np.exp(alpha[a]-beta[h]) * mu_a
        if k_disp and k_disp < 1e6:
            lh = rng.gamma(k_disp, lh/k_disp); la = rng.gamma(k_disp, la/k_disp)
        gh = rng.poisson(lh); ga = rng.poisson(la)
        if gh > ga: return h
        if ga > gh: return a
        ph = ppd.get(h,0.5); pa = ppd.get(a,0.5)
        pn = ph/(ph+pa) if (ph+pa)>0 else 0.5
        return h if rng.random() < pn else a

    def play_goals(h, a):
        lh = np.exp(alpha[h]-beta[a]) * mu_h; la = np.exp(alpha[a]-beta[h]) * mu_a
        if k_disp and k_disp < 1e6:
            lh = rng.gamma(k_disp, lh/k_disp); la = rng.gamma(k_disp, la/k_disp)
        return rng.poisson(lh), rng.poisson(la)

    for s in range(n_sims):
        winners = {}; runners = {}; thirds = []
        for g in group_list:
            ts = groups[g]
            pts = {t:0 for t in ts}; gd = {t:0 for t in ts}; gf = {t:0 for t in ts}
            for i,j in gfix:
                h, a = ts[i], ts[j]
                if (h, a) in played_scores:
                    gh, ga = played_scores[(h, a)]
                elif (a, h) in played_scores:
                    ga, gh = played_scores[(a, h)]
                else:
                    gh, ga = play_goals(h, a)
                gf[h]+=gh; gf[a]+=ga; gd[h]+=gh-ga; gd[a]+=ga-gh
                if gh>ga: pts[h]+=3
                elif ga>gh: pts[a]+=3
                else: pts[h]+=1; pts[a]+=1
            rank = sorted(ts, key=lambda t:(pts[t],gd[t],gf[t]), reverse=True)
            winners[g]=rank[0]; runners[g]=rank[1]
            cnt[rank[0]]["1st"]+=1; cnt[rank[1]]["2nd"]+=1; cnt[rank[2]]["3rd"]+=1
            cnt[rank[0]]["adv"]+=1; cnt[rank[1]]["adv"]+=1
            thirds.append((pts[rank[2]],gd[rank[2]],gf[rank[2]],g,rank[2]))

        thirds.sort(reverse=True)
        best8 = thirds[:8]
        best8_groups = {g for (_,_,_,g,_) in best8}
        third_by_group = {g:t for (_,_,_,g,t) in best8}
        for (_,_,_,_,t) in best8: cnt[t]["adv"]+=1

        def resolve(slot):
            typ, grp = slot.split(":")
            if typ=="W": return winners[grp]
            if typ=="R": return runners[grp]
            for g in grp:
                if g in best8_groups and third_by_group.get(g) is not None:
                    t = third_by_group[g]; third_by_group[g]=None
                    best8_groups.discard(g); return t
            for g in list(best8_groups):
                t = third_by_group.get(g)
                if t: third_by_group[g]=None; best8_groups.discard(g); return t
            return None

        win = {}
        for m,(sa,sb) in BR.R32.items():
            w = play(resolve(sa), resolve(sb)); win[m]=w; cnt[w]["r16"]+=1
        for m,(a,b) in BR.R16.items():
            w = play(win[a],win[b]); win[m]=w; cnt[w]["qf"]+=1
        for m,(a,b) in BR.QF.items():
            w = play(win[a],win[b]); win[m]=w; cnt[w]["sf"]+=1
        for m,(a,b) in BR.SF.items():
            w = play(win[a],win[b]); win[m]=w; cnt[w]["final"]+=1
        champ = play(win[BR.FINAL[0]], win[BR.FINAL[1]])
        cnt[champ]["champ"]+=1

    rows = []
    for t in all_teams:
        c = cnt[t]
        rows.append({"team":t,
            "p_1st":c["1st"]/n_sims,"p_2nd":c["2nd"]/n_sims,"p_3rd":c["3rd"]/n_sims,
            "p_advance":c["adv"]/n_sims,"p_r16":c["r16"]/n_sims,"p_qf":c["qf"]/n_sims,
            "p_sf":c["sf"]/n_sims,"p_final":c["final"]/n_sims,"p_champ":c["champ"]/n_sims})
    return pd.DataFrame(rows).sort_values("p_champ", ascending=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--shrink", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = simulate(args.sims, args.k, args.shrink, args.seed)
    print(df.head(15).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False); print(f"\nGuardado: {args.out}")
