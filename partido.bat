@echo off
REM ============================================================
REM  partido.bat - Forescore Mundial 2026
REM
REM  Resultado normal:
REM    partido.bat "Mexico" "South Africa" 2-1
REM
REM  Registrar clasificado (al cerrar un grupo):
REM    partido.bat qualify "Mexico" "W:A"
REM    partido.bat qualify "Argentina" "R:B"
REM    partido.bat qualify "Morocco" "T:ABCDF"
REM ============================================================

REM -- Limitar hilos de BLAS: evita el OpenBLAS OOM en la simulacion --
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1

REM -- Modo qualify --
if /i "%~1"=="qualify" (
    if "%~3"=="" (
        echo.
        echo  USO: partido.bat qualify "Equipo" "Slot"
        echo  Slots: W:X ^(1ro grupo X^), R:X ^(2do grupo X^), T:XXXX ^(mejor tercero^)
        echo  Ejemplo: partido.bat qualify "Mexico" "W:A"
        echo.
        exit /b 1
    )
    set TEAM=%~2
    set SLOT=%~3
    echo.
    echo ============================================================
    echo   Registrando clasificado: %TEAM% -- %SLOT%
    echo ============================================================
    echo.
    python live_update.py qualify --team "%TEAM%" --slot "%SLOT%"
    if errorlevel 1 ( echo ERROR en qualify. Abortando. & exit /b 1 )
    echo.
    echo [2/2] Subiendo a GitHub...
    git add forescore_mundial_dashboard.html live_state.json
    git commit -m "Clasificado: %TEAM% %SLOT%"
    git push
    echo.
    echo ============================================================
    echo   LISTO. %TEAM% aparece confirmado en el bracket.
    echo ============================================================
    echo.
    exit /b 0
)

REM -- Modo resultado normal --
if "%~3"=="" (
    echo.
    echo  USO: partido.bat "Local" "Visitante" Marcador
    echo  Ejemplo: partido.bat "Mexico" "South Africa" 2-1
    echo.
    exit /b 1
)

set HOME_TEAM=%~1
set AWAY_TEAM=%~2
set SCORE=%~3

echo.
echo ============================================================
echo   Procesando: %HOME_TEAM% %SCORE% %AWAY_TEAM%
echo ============================================================
echo.

echo [1/4] Aplicando resultado (Bayesian update)...
python live_update.py update --home "%HOME_TEAM%" --away "%AWAY_TEAM%" --score %SCORE%
if errorlevel 1 (
    echo ERROR en update. Abortando.
    exit /b 1
)

echo.
echo [2/4] Re-simulando torneo (50.000 sims, puede tardar ~1 min)...
python live_update.py resimulate --sims 50000
if errorlevel 1 (
    echo ERROR en resimulate. Abortando.
    exit /b 1
)

echo.
echo [3/4] Guardando cambios en git...
git add forescore_mundial_dashboard.html live_state.json mc_final.csv
git commit -m "%HOME_TEAM% %SCORE% %AWAY_TEAM%"

echo.
echo [4/4] Subiendo a GitHub...
git push

echo.
echo ============================================================
echo   LISTO. Tus amigos ya ven el dashboard actualizado.
echo   (GitHub Pages tarda ~1 min en refrescar)
echo ============================================================
echo.
