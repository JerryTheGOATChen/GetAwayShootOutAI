@echo off
rem Trains both methods back to back (so they don't share the CPU), then draws the comparison chart.
rem   train_all.bat            start both from scratch
rem   train_all.bat --resume   carry on from the last checkpoints
cd /d "%~dp0"
set "PY=%LocalAppData%\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" -u -m ppo.train --steps 50e6 %*
if errorlevel 1 goto failed
"%PY%" -u -m genetic.train --steps 50e6 %*
if errorlevel 1 goto failed
"%PY%" -m common.compare
if errorlevel 1 goto failed

echo.
echo Training finished. Chart: runs\compare.png
goto end

:failed
echo.
echo Training stopped before finishing. Progress is saved; run "train_all.bat --resume" to continue.

:end
