# Forescore Mundial 2026 · v3.1

Modelo predictivo del Mundial 2026: **Dixon-Coles + Monte Carlo (50.000 sims)** con bracket oficial del torneo y capas de calibración.

🔗 **Dashboard:** `https://TU-USUARIO.github.io/forescore-mundial-2026/forescore_mundial_dashboard.html`

## Qué hace

- Probabilidades de pasar grupos, R32, R16, cuartos, semis, final y campeón para las 48 selecciones.
- Predicción partido a partido (λ, 1X2, top-5 marcadores) con ajustes contextuales.
- Bayesian update post-partido y re-simulación del torneo completo.

## El modelo (v3.1)

Sobre la base Dixon-Coles (ρ = −0.092) se aplican:

1. **Bracket oficial Mundial 2026** — los 16 cruces R32 (matches 73-88) con clusters de terceros y el árbol completo hasta la final. Fue el cambio más importante: el Monte Carlo original sobreestimaba a los favoritos (Argentina 22%) por no usar el bracket real; con él baja a ~13%, más realista.
2. **Penales por equipo** — probabilidad histórica de tanda (Alemania 79%, Argentina 75%, España 36%...).
3. **Tendencia al empate por equipo** — corrección empírica (Suiza +21%, Argentina −13%...).
4. **Varianza inflada** (Gamma-Poisson, k=18) — ensancha la incertidumbre para no sobreconcentrar en favoritos.
5. **Shrinkage por confederación** (λ=0.15) — regulariza equipos con datos ruidosos (Curazao, Jordania).
6. **Anclaje Elo** (w=0.25) — mezcla la fuerza del modelo con el Elo de selecciones (snapshot pre-Mundial 27-may-2026), corrigiendo la compresión de los favoritos.

## Comandos

### Predicción pre-partido
```bash
python live_update.py predict --home España --away "Arabia Saudi" --absences-home "Pedri"
```

### Update post-partido (regenera dashboard de marcadores)
```bash
python live_update.py update --home España --away "Arabia Saudi" --score 3-0
```

### Re-simular torneo (actualiza P(campeón) con lo aprendido)
```bash
python live_update.py resimulate
```
Re-ancla a Elo, re-corre 50.000 simulaciones con los α/β actuales (base + ajustes de los `update`) y reescribe las probabilidades de campeón/fases. **Esto es lo que hace que las probabilidades de campeón cambien durante el torneo.**

### Otros
```bash
python live_update.py status              # ajustes acumulados
python live_update.py reset               # volver al modelo base
python live_update.py rebuild-dashboard   # regenerar HTML de marcadores
```

## Flujo durante el torneo

```
Tras cada partido:
    python live_update.py update --home X --away Y --score 2-1
    git add forescore_mundial_dashboard.html live_state.json
    git commit -m "X 2-1 Y" && git push

Al final de cada fase (grupos, R32, R16...):
    python live_update.py resimulate
    git add forescore_mundial_dashboard.html && git commit -m "Re-sim tras grupos" && git push
```

## Estructura

```
├── live_update.py            # CLI principal
├── monte_carlo.py            # Motor de simulación (bracket oficial)
├── bracket_2026.py           # Estructura oficial del bracket
├── elo_anchor.py             # Anclaje de α/β hacia Elo
├── apply_simulation.py       # Inyecta resultados MC en el dashboard
├── build_dashboard.py        # Regenera HTML de marcadores
├── dashboard_template.html   # Plantilla base
├── forescore_mundial_dashboard.html  # Dashboard final (el que se publica)
├── dixon_coles_params.csv    # α, β base
├── dixon_coles_params_elo.csv# α, β anclados a Elo (generado)
├── elo_base.csv              # Elo pre-Mundial de los 48
├── mc_final.csv              # Última simulación
├── penalty_probs.csv / draw_tendency.csv / optimized_params.csv
└── teams_base.json / matches_base.json
```

`live_state.json` se crea localmente al primer update (en `.gitignore`).

## Parámetros calibrables

- `--k` (varianza): bajo = más sorpresas, alto = más determinista. Default 18.
- `--shrink` (regularización confederación): 0 a 1. Default 0.15.
- `--elo-w` (peso Elo): 0 a 1. Default 0.25. Por encima de 0.40 el modelo se vuelve un repetidor del Elo.

## Limitaciones (honestas)

- Los α/β se recuperaron de las λ del dashboard original por mínimos cuadrados, no de datos crudos. El anclaje Elo corrige parte de la compresión resultante.
- El modelo no usa xG (entrena con goles), no modela cambios de seleccionador, lesiones intra-partido, árbitros ni VAR.
- El bracket de terceros usa una asignación válida de los clusters FIFA, no las 495 combinaciones exactas (efecto despreciable en agregado).
- **No está validado contra un torneo real con esta configuración.** Corrige sesgos conocidos pero es "el mejor esfuerzo con los datos disponibles", no un modelo demostrablemente óptimo.

---

Forescore Mundial 2026 · v3.1 · Motor Monte Carlo con bracket oficial + Elo
