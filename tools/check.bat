@echo off
REM Local pre-commit check: byte-compile + full smoke test (uses the venv).
cd /d "%~dp0\.."
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3.10"
echo === py_compile ===
%PY% -m py_compile app.py tools\smoke_test.py tools\check_zimage_models.py || (echo COMPILE FAILED & exit /b 1)
echo === config JSON ===
%PY% -c "import json; json.load(open('config-sample.txt',encoding='utf-8')); print('config-sample.txt OK')" || exit /b 1
echo === smoke test ===
%PY% tools\smoke_test.py
