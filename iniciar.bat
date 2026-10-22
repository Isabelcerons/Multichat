@echo off
chcp 65001 >nul
title Multichat
REM ==================================================================
REM  Multichat - doble clic en este archivo.
REM  La primera vez instala todo (librerias, llama.cpp, EmbeddingGemma,
REM  modelos de Ollama). Las siguientes veces arranca en segundos.
REM ==================================================================
cd /d "%~dp0"
if not exist tools mkdir tools
if not exist models mkdir models

REM ---------------- Python ----------------
set PY=
where python >nul 2>nul && set PY=python
if not defined PY ( where py >nul 2>nul && set PY=py )
if not defined PY (
  echo [ERROR] No encontre Python. Instalalo desde https://www.python.org/downloads/ y marca "Add Python to PATH".
  goto :fin
)
if not exist .venv\Scripts\python.exe (
  echo === Primera vez: creando el entorno de Python ===
  %PY% -m venv .venv || goto :error
)
if not exist .venv\instalado-v1.txt (
  echo Instalando librerias - unos minutos la primera vez...
  .venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements.txt || goto :error
  echo ok> .venv\instalado-v1.txt
)

REM ---------------- Configuracion: contrasena de PostgreSQL ----------------
if not exist .env (
  copy .env.example .env >nul
  echo.
  echo  Se abrira el archivo .env: cambia TU_CONTRASENA por la contrasena de PostgreSQL,
  echo  guarda con Ctrl+G y cierra el Bloc de notas.
  echo.
  notepad .env
)

REM ---------------- Ollama: modelo que responde y modelo de vision ----------------
where ollama >nul 2>nul || (
  echo [AVISO] No encontre Ollama: instalalo desde https://ollama.com/download o usa Gemini en Ajustes.
  goto :llama
)
powershell -NoProfile -Command "try{Invoke-WebRequest http://localhost:11434 -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0}catch{exit 1}" || (
  echo Encendiendo Ollama...
  start "" /min ollama serve
  powershell -NoProfile -Command "for($i=0;$i -lt 30;$i++){try{Invoke-WebRequest http://localhost:11434 -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0}catch{Start-Sleep 1}}; exit 1"
)
if not exist .venv\modelos-v1.txt (
  echo Revisando modelos de Ollama...
  ollama list > .venv\lista.txt 2>nul
  findstr /i "qwen2.5:1.5b" .venv\lista.txt >nul || ollama pull qwen2.5:1.5b
  findstr /i "moondream" .venv\lista.txt >nul || ollama pull moondream
  echo ok> .venv\modelos-v1.txt
)

:llama
REM ---------------- llama.cpp: servidor de embeddings con EmbeddingGemma ----------------
set LLAMA=
for /f "delims=" %%f in ('dir /s /b tools\llama\llama-server.exe 2^>nul') do set "LLAMA=%%f"
if not defined LLAMA (
  echo Descargando llama.cpp - version CPU para Windows...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; try { $r = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -Headers @{'User-Agent'='multichat'}; $ok = $r.assets | Where-Object { $_.name -like '*win*x64*.zip' -and $_.name -notmatch 'cuda|cudart|vulkan|sycl|hip|kompute|opencl|arm' }; $a = ($ok | Where-Object { $_.name -like '*cpu*' } | Select-Object -First 1); if (-not $a) { $a = $ok | Select-Object -First 1 }; if (-not $a) { Write-Host 'Archivos disponibles:'; $r.assets | ForEach-Object { Write-Host (' - ' + $_.name) }; throw 'no hay un zip para Windows x64' }; Write-Host ('Descargando ' + $a.name); curl.exe -L --fail -o 'tools\llama.zip' $a.browser_download_url; Expand-Archive 'tools\llama.zip' 'tools\llama' -Force; Remove-Item 'tools\llama.zip'; Write-Host ('llama.cpp ' + $r.tag_name + ' listo') } catch { Write-Host ('[AVISO] No se pudo descargar llama.cpp: ' + $_) }"
  for /f "delims=" %%f in ('dir /s /b tools\llama\llama-server.exe 2^>nul') do set "LLAMA=%%f"
)
set GGUF=models\embeddinggemma-300M-Q8_0.gguf
if not exist %GGUF% (
  echo Descargando EmbeddingGemma-300M en formato GGUF - unos 330 MB...
  curl.exe -L --fail -o %GGUF%.part "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf" && move /y %GGUF%.part %GGUF% >nul
  if exist %GGUF%.part del %GGUF%.part
)
set USE_LLAMA=
if defined LLAMA if exist %GGUF% set USE_LLAMA=1
if defined USE_LLAMA (
  powershell -NoProfile -Command "try{Invoke-WebRequest http://localhost:8081/health -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0}catch{exit 1}" || (
    echo Encendiendo llama.cpp con EmbeddingGemma en el puerto 8081...
    start "llama.cpp - EmbeddingGemma" /min "%LLAMA%" -m %GGUF% --embeddings --port 8081 -c 2048 -b 2048 -ub 2048 --threads 2
    powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest http://localhost:8081/health -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0}catch{Start-Sleep 1}}; exit 1" || echo [AVISO] llama.cpp no respondio a tiempo.
  )
) else (
  echo [AVISO] No se pudo preparar llama.cpp: se usara EmbeddingGemma desde Ollama.
  where ollama >nul 2>nul && ollama pull embeddinggemma
)

REM ---------------- Tesseract: idioma espanol para leer texto en imagenes ----------------
if not exist tessdata\spa.traineddata (
  mkdir tessdata 2>nul
  powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; $u='https://github.com/tesseract-ocr/tessdata_fast/raw/main/'; foreach($l in 'spa','eng','osd'){ try{ Invoke-WebRequest ($u+$l+'.traineddata') -OutFile ('tessdata\'+$l+'.traineddata') -UseBasicParsing }catch{} }"
)

REM ---------------- Ejemplos: crea los audios y videos de muestra la primera vez ----------------
if not exist .venv\ejemplos-v1.txt (
  echo Creando los audios y videos de ejemplo con la voz de Windows...
  powershell -NoProfile -ExecutionPolicy Bypass -File ejemplos\crear_ejemplos.ps1
  echo ok> .venv\ejemplos-v1.txt
)

REM ---------------- Si ya esta abierto, solo abre la pagina ----------------
powershell -NoProfile -Command "try{Invoke-WebRequest http://localhost:8000/api/index -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0}catch{exit 1}" && (
  start "" http://localhost:8000
  goto :fin
)

echo.
echo ===  Multichat en http://localhost:8000  ===
echo      Deja esta ventana abierta mientras lo usas. Para cerrar: Ctrl+C
echo.
start "" "%~dp0cargando.html"
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
goto :fin

:error
echo.
echo [ERROR] Algo fallo durante la instalacion. Toma una captura de esta ventana.

:fin
echo.
pause
