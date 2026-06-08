# Forescore Mundial 2026 · v3.0

Modelo predictivo del Mundial 2026 basado en **Dixon-Coles** + Monte Carlo (50.000 simulaciones), con tres mejoras sobre el modelo base:

1. **Penales por equipo** (datos históricos de Mundiales/Euros/Copas continentales 1976-2024)
2. **Tendencia al empate por equipo** (calibrada sobre 2.500+ partidos)
3. **Threshold óptimo 1X2** (P(empate)≥0.38 ∧ |P(H)−P(A)|≤0.26)

🔗 **Dashboard en vivo:** _(actívalo siguiendo los pasos de GitHub Pages)_

## Quick start

```bash
git clone https://github.com/TU-USUARIO/forescore-mundial-2026.git
cd forescore-mundial-2026
pip install -r requirements.txt
open forescore_mundial_dashboard.html
```

## Comandos

### Predicción pre-partido

```bash
python live_update.py predict \
    --home Argentina --away Argelia \
    --absences-away "Mahrez,Bensebaini" \
    --backup-gk-away
```

Acepta nombres en español o inglés. Aplica los tres ajustes del modelo v3 automáticamente.

| Flag | Efecto |
|---|---|
| `--absences-home "J1,J2"` / `--absences-away` | −10% λ ofensiva por jugador (cap 50%) |
| `--backup-gk-home` / `--backup-gk-away` | +8% λ del rival |
| `--rotation-home` / `--rotation-away` | −5% λ propia |
| `--weather-extreme` | −5% λ ambos |
| `--home-country` | aplica γ (solo MEX/USA/CAN en sus sedes) |
| `--knockout` | muestra prob. de tanda de penales |

### Update post-partido (regenera dashboard automáticamente)

```bash
python live_update.py update --home Argentina --away Argelia --score 3-1
```

Aplica Bayesian update (peso 0.30) sobre α/β y regenera `forescore_mundial_dashboard.html`:
- Partidos jugados → marcados en verde con resultado real
- Partidos pendientes → λ recalculadas con ajustes acumulados
- Banner superior con resumen del estado live

### Otros

```bash
python live_update.py status              # ver ajustes acumulados
python live_update.py reset               # volver al modelo base
python live_update.py rebuild-dashboard   # regenerar HTML sin update
```

## Estructura del repo

```
├── live_update.py              # CLI principal
├── build_dashboard.py          # Regeneración del HTML
├── extract_and_fit.py          # Inicialización (ya ejecutado, no hace falta correr)
├── dashboard_template.html     # Plantilla base (v3 frozen)
├── forescore_mundial_dashboard.html  # Dashboard auto-actualizable
├── dixon_coles_params.csv      # α, β por equipo
├── dixon_coles_global.csv      # γ, ρ
├── penalty_probs.csv           # P(ganar tanda) por equipo
├── draw_tendency.csv           # Desviación empírica del empate por equipo
├── optimized_params.csv        # threshold_pd, threshold_diff, draw_strength
├── teams_base.json             # Probabilidades pre-Mundial Monte Carlo
├── matches_base.json           # 72 partidos de grupos con λ
└── requirements.txt
```

`live_state.json` se crea en cada update local y está en `.gitignore`.

## Modelo

- **Dixon-Coles**: Poisson bivariado con corrección ρ = −0.092 para marcadores bajos
- **α** (ataque), **β** (vulnerabilidad defensiva) por equipo
- **γ** (ventaja de local) muy baja (0.04): casi todos los partidos en sede neutral; solo aplica con `--home-country`
- **Penales eliminatorias**: probabilidad histórica por equipo (Alemania 79%, Argentina 75%, España 36%...). Resto: prior 50%.
- **Ajuste empate**: corrección empírica por equipo (Suiza +20.9%, Argentina −12.6%...)
- **Threshold 1X2**: predice empate si P(empate)≥0.38 ∧ |P(H)−P(A)|≤0.26
- **Bayesian update post-partido**: peso 0.30 sobre log(λ_real) vs log(λ_predicha)

## Publicar con GitHub Pages

1. Sube el repo a GitHub (público).
2. **Settings → Pages → Source**: `Deploy from a branch` → `main` / `(root)` → Save.
3. En ~1 min disponible en:
   `https://TU-USUARIO.github.io/forescore-mundial-2026/forescore_mundial_dashboard.html`
4. Tras cada update local:
   ```bash
   git add forescore_mundial_dashboard.html
   git commit -m "Tras X-Y"
   git push
   ```

## Limitaciones

- No modela cambios de seleccionador
- No modela lesiones intra-partido
- No incluye árbitros, VAR ni xG histórico
- `penalty_probs.csv` solo tiene los 9 equipos del reporte; el resto usa prior 50%. Si quieres añadir más, edita el CSV.
- `draw_tendency.csv` solo tiene los 5 equipos del reporte; el resto neutro.

---

Forescore Mundial 2026 · v3.0 · Anthropic Claude
