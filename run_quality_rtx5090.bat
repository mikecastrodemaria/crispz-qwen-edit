@echo off
title crispz-qwen-edit - RTX 5090 (local)
cd /d "%~dp0"
echo ============================================
echo  crispz-qwen-edit - RTX 5090 (local 127.0.0.1)
echo ============================================
echo.
REM Optimisations CUDA (sans danger, BF16)
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set GRADIO_SERVER_PORT=7860
REM Console UTF-8 (evite les crashs cp1252 sur les barres de progression HF)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM === LOCAL-ONLY: utilise UNIQUEMENT le cache HF, ne RE-telecharge jamais Qwen-Image. ===
REM Qwen/Qwen-Image est deja en cache (46 Go) -> charge en local, 0 telechargement.
REM Effet de bord: l'onglet Edit (Qwen-Image-Edit-2509, non cache) affichera une erreur
REM au lieu d'aspirer ~20 Go. Mets cette ligne en commentaire (REM) pour autoriser les
REM telechargements une fois (puis remets-la).
set HF_HUB_OFFLINE=1
REM Delegue au run.bat (detection venv + ESRGAN_DIR + lancement)
call "%~dp0run.bat" %*
