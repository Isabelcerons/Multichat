@echo off
chcp 65001 >nul
title Multichat - crear ejemplos
cd /d "%~dp0"
echo Creando la base de conocimiento de ejemplo (PDF, imagen, audios y videos)...
powershell -NoProfile -ExecutionPolicy Bypass -File ejemplos\crear_ejemplos.ps1
echo.
echo Ahora abre Multichat (iniciar.bat) y pulsa "Indexar carpeta".
pause
