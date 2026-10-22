# Genera los audios y videos de ejemplo de la base de conocimiento con la voz en espanol de Windows
# y ffmpeg. Se ejecuta desde crear_ejemplos.bat
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$kb = Join-Path $root 'conocimiento'
$tmp = Join-Path $env:TEMP 'multichat_ejemplos'
New-Item -ItemType Directory -Force -Path $tmp, "$kb\audios", "$kb\videos", "$kb\pdfs", "$kb\imagenes" | Out-Null

# --- ffmpeg: en el PATH o en la carpeta del usuario ---
$ff = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ff) { $ff = Get-ChildItem "$env:USERPROFILE\ffmpeg" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName }
if (-not $ff) { Write-Host '[ERROR] No encontre ffmpeg. Descargalo de https://www.gyan.dev/ffmpeg/builds/'; exit 1 }
Write-Host "ffmpeg: $ff"

# --- voz en espanol ---
Add-Type -AssemblyName System.Speech
$voz = New-Object System.Speech.Synthesis.SpeechSynthesizer
$es = $voz.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'es*' } | Select-Object -First 1
if ($es) { $voz.SelectVoice($es.VoiceInfo.Name); Write-Host "Voz: $($es.VoiceInfo.Name)" }
else { Write-Host '[AVISO] Windows no tiene voz en espanol instalada: se usara la voz por defecto.' }
$voz.Rate = 0

function Hablar($texto, $wav) {
  $voz.SetOutputToWaveFile($wav)
  $voz.Speak($texto)
  $voz.SetOutputToNull()
}

# --- audios ---
Get-ChildItem "$PSScriptRoot\_audios\*.txt" | ForEach-Object {
  $wav = Join-Path $tmp ($_.BaseName + '.wav')
  Hablar (Get-Content $_.FullName -Raw -Encoding UTF8) $wav
  & $ff -loglevel error -y -i $wav -ac 1 -b:a 64k (Join-Path "$kb\audios" ($_.BaseName + '.mp3'))
  Write-Host "Audio creado: $($_.BaseName).mp3"
}

# --- videos: una diapositiva + su narracion por segmento, luego se unen ---
$grupos = Get-ChildItem "$PSScriptRoot\_videos\*.png" | Group-Object { $_.BaseName -replace '_\d+$', '' }
foreach ($g in $grupos) {
  $lista = Join-Path $tmp "$($g.Name).txt"
  Remove-Item $lista -ErrorAction SilentlyContinue
  foreach ($png in ($g.Group | Sort-Object Name)) {
    $wav = Join-Path $tmp ($png.BaseName + '.wav')
    $seg = Join-Path $tmp ($png.BaseName + '.mp4')
    Hablar (Get-Content ($png.FullName -replace '\.png$', '.txt') -Raw -Encoding UTF8) $wav
    & $ff -loglevel error -y -loop 1 -i $png.FullName -i $wav -shortest -c:v libx264 -tune stillimage -pix_fmt yuv420p -r 10 -c:a aac -b:a 96k $seg
    Add-Content $lista ("file '" + $seg.Replace('\', '/') + "'")
  }
  & $ff -loglevel error -y -f concat -safe 0 -i $lista -c copy (Join-Path "$kb\videos" ($g.Name + '.mp4'))
  Write-Host "Video creado: $($g.Name).mp4"
}

# --- PDFs e imagen de ejemplo ---
Copy-Item "$PSScriptRoot\pdfs\*.pdf" "$kb\pdfs" -Force
Copy-Item "$PSScriptRoot\_videos\video_carga_carro_electrico_1.png" "$kb\imagenes\foto_carro_semillero.png" -Force
Write-Host ''
Write-Host 'Listo: la carpeta conocimiento tiene PDFs, una imagen, 2 audios y 2 videos.'
