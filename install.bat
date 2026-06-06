@echo off
REM Install pour crispz - Z-Image upscaler + detailer (Windows)
REM Defaut: venv .venv (--system-site-packages) qui HERITE de ton torch et isole
REM les deps. --no-venv (ou --system) pour installer sur le Python courant.
REM Ne reinstalle JAMAIS torch.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- flags ---
set USE_VENV=1
set FACESWAP=1
set FACESWAP_MODEL=0
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-venv" set USE_VENV=0
if /I "%~1"=="--system" set USE_VENV=0
if /I "%~1"=="--no-faceswap" set FACESWAP=0
if /I "%~1"=="--faceswap-model" set FACESWAP_MODEL=1
shift
goto argloop
:argdone

echo === crispz-studio - install Windows ===
echo.

REM 1) Python de base
where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] Python introuvable. Installe Python 3.10+ depuis python.org.
        exit /b 1
    )
    set PYCMD=python
) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)
echo Python de base: !PYCMD!
!PYCMD! --version
echo.

REM 2) torch + CUDA (NE PAS reinstaller)
!PYCMD! -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 2 (
    echo.
    echo [AVERT] PyTorch present mais CUDA non disponible. Z-Image en CPU sera tres lent.
    goto torch_ok
)
if errorlevel 1 (
    echo.
    echo [ERREUR] PyTorch introuvable. Installe d'abord ton build PyTorch + CUDA, puis relance.
    echo Exemple ^(CUDA 12.8^): !PYCMD! -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    exit /b 1
)
:torch_ok
echo.

REM 3) xformers casse ? le neutraliser cote SYSTEME (le venv en herite)
!PYCMD! -c "import xformers.ops" >nul 2>&1
if not errorlevel 1 (
    echo xformers OK.
) else (
    !PYCMD! -c "import xformers" >nul 2>&1
    if not errorlevel 1 (
        echo [AVERT] xformers installe mais ne charge pas ^(DLL/ABI torch incompatible^). Desinstallation.
        !PYCMD! -m pip uninstall -y xformers
    )
)
echo.

REM 4) venv optionnel (defaut) avec --system-site-packages
set RUNPY=!PYCMD!
if "!USE_VENV!"=="1" (
    if not exist ".venv\Scripts\python.exe" (
        echo Creation du venv .venv ^(--system-site-packages: herite de torch^)...
        !PYCMD! -m venv --system-site-packages .venv
    )
    if exist ".venv\Scripts\python.exe" (
        .venv\Scripts\python.exe -c "import torch" >nul 2>&1
        if errorlevel 1 (
            echo [AVERT] torch non visible dans le venv -^> repli sur le Python courant.
        ) else (
            set RUNPY=.venv\Scripts\python.exe
        )
    )
) else (
    echo Mode --no-venv: install sur le Python courant.
)
echo Interpreteur d'install: !RUNPY!
echo.

REM 5) Installer les deps
echo Installation des dependances...
!RUNPY! -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERREUR] echec pip install. Verifie le log ci-dessus.
    exit /b 1
)
echo.

REM 6) Verifier ZImageImg2ImgPipeline
!RUNPY! -c "from diffusers import ZImageImg2ImgPipeline; print('ZImageImg2ImgPipeline OK')"
if errorlevel 1 (
    echo [ERREUR] diffusers ne contient pas ZImageImg2ImgPipeline.
    exit /b 1
)
echo.

REM 7) Deps FaceSwap (optionnelles, par defaut ON ; --no-faceswap pour sauter)
if "!FACESWAP!"=="1" (
    echo Installation des deps FaceSwap ^(insightface + onnxruntime-gpu^)...
    !RUNPY! -m pip install -r requirements-faceswap.txt
    if errorlevel 1 echo [AVERT] echec install FaceSwap ^(non bloquant^). La feature restera desactivee.
    echo.
)

REM 8) Dossiers de modeles
for %%D in (upscale_models checkpoints loras faceswap) do if not exist "%%D" mkdir "%%D"
echo Dossiers prets: upscale_models (ESRGAN), checkpoints (Z-Image), loras, faceswap.
echo.

REM 9) Config locale: copie config-sample.txt -> config.txt si absent
if not exist "config.txt" (
    if exist "config-sample.txt" (
        copy /Y "config-sample.txt" "config.txt" >nul
        echo config.txt cree depuis config-sample.txt ^(edite-le pour tes reglages^).
    )
)
echo.

REM 10) Modele inswapper (FaceSwap) - opt-in ^(528 Mo, licence^): --faceswap-model
if "!FACESWAP_MODEL!"=="1" (
    if not exist "faceswap\inswapper_128.onnx" (
        echo Telechargement du modele inswapper_128.onnx ^(~528 Mo^)...
        !RUNPY! -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', 'faceswap/inswapper_128.onnx'); print('inswapper OK')"
    ) else (
        echo Modele inswapper deja present.
    )
    echo.
)

echo === Install OK. Lance run.bat  ^(ou run.bat --no-venv^) ===
echo     Options install: --no-faceswap ^(sauter insightface^)  --faceswap-model ^(telecharger inswapper^)
endlocal
