@echo off
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
  echo Python is missing. Install Python 3.11 from the page that just opened.
  echo Tick "Add python.exe to PATH", then double-click this file again.
  start https://www.python.org/downloads/windows/
  pause
  exit /b 1
)
py -3 -m pip install --upgrade pip
py -3 -m pip install numpy pandas pyarrow scikit-learn lightgbm
if errorlevel 1 (
  echo Install failed.
  pause
  exit /b 1
)
if not exist results mkdir results
for %%F in (1 2 3 4 5 6 7 8 9) do (
  start "lightgbm fold %%F" cmd /c "py -3 -u experiment.py --data features.parquet --results results --algo lightgbm --fold %%F --tree-jobs 8 1> results\fold%%F.log 2>&1 & echo fold %%F finished & pause"
)
echo Started folds 1 to 9. Stay signed in until every window says finished.
pause