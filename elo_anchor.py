"""
elo_anchor.py — Ancla los alpha/beta del modelo hacia la fuerza Elo.

Idea:
  - La "fuerza neta" de un equipo en el modelo es s_model = alpha - beta.
  - El Elo da otra medida de fuerza s_elo (estandarizada).
  - Hacemos blend: s_final = (1-w)*s_model + w*s_elo_mapeado
  - Reconstruimos alpha/beta preservando la suma alpha+beta (capacidad de
    'producir goles' del partido) pero ajustando la diferencia alpha-beta
    (quien domina). Asi el Elo mueve QUIEN gana sin inflar el total de goles.

Parametro:
  w = peso del Elo en [0,1]. w=0 -> modelo puro; w=1 -> fuerza puramente Elo.

Uso:
  python elo_anchor.py --w 0.25        # genera dixon_coles_params_elo.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

WORKDIR = Path(__file__).parent

def build(w=0.25, verbose=True):
    dc = pd.read_csv(WORKDIR/"dixon_coles_params.csv")
    elo = pd.read_csv(WORKDIR/"elo_base.csv")
    m = dc.merge(elo[["team","rating"]], on="team", how="left")
    if m["rating"].isna().any():
        missing = m[m["rating"].isna()]["team"].tolist()
        raise SystemExit(f"Faltan Elo para: {missing}")

    # Fuerza del modelo: s_model = alpha - beta
    m["s_model"] = m["alpha"] - m["beta"]
    # suma (nivel de goles del equipo) que queremos preservar
    m["sum_ab"] = m["alpha"] + m["beta"]

    # Estandarizar ambas escalas a media 0, desv 1
    sm_mean, sm_std = m["s_model"].mean(), m["s_model"].std()
    el_mean, el_std = m["rating"].mean(), m["rating"].std()
    m["z_model"] = (m["s_model"] - sm_mean) / sm_std
    m["z_elo"]   = (m["rating"]  - el_mean) / el_std

    # Blend en espacio z, luego devolver a escala de s_model
    m["z_blend"] = (1-w)*m["z_model"] + w*m["z_elo"]
    m["s_new"] = m["z_blend"] * sm_std + sm_mean

    # Reconstruir alpha/beta preservando la suma (sum_ab):
    #   alpha_new = (sum_ab + s_new)/2 ; beta_new = (sum_ab - s_new)/2
    m["alpha_new"] = (m["sum_ab"] + m["s_new"]) / 2
    m["beta_new"]  = (m["sum_ab"] - m["s_new"]) / 2

    out = m[["team","alpha_new","beta_new"]].rename(
        columns={"alpha_new":"alpha","beta_new":"beta"})
    out.to_csv(WORKDIR/"dixon_coles_params_elo.csv", index=False)

    if verbose:
        # Mostrar mayores movimientos
        m["delta_s"] = m["s_new"] - m["s_model"]
        big = m.reindex(m["delta_s"].abs().sort_values(ascending=False).index)
        print(f"Blend Elo w={w}. Mayores cambios en fuerza neta (alpha-beta):")
        print(f"{'Equipo':<16}{'s_model':>9}{'s_elo_z':>9}{'s_new':>8}{'Δ':>7}")
        for _,r in big.head(12).iterrows():
            print(f"{r['team']:<16}{r['s_model']:>9.3f}{r['z_elo']:>9.2f}{r['s_new']:>8.3f}{r['delta_s']:>+7.3f}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=0.25, help="peso del Elo [0-1]")
    args = ap.parse_args()
    build(args.w)
    print("\nGuardado: dixon_coles_params_elo.csv")
