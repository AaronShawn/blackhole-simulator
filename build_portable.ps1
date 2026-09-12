# Rebuild the portable Windows package (dist\BlackHoleSimulator).
#
#   powershell -ExecutionPolicy Bypass -File build_portable.ps1
#
# Requirements: python 3.9+, pip install pywebview pyinstaller pillow

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/3] generating icon ..."
python tools\make_icon.py

Write-Host "[2/3] building with PyInstaller ..."
pyinstaller --noconfirm --clean --windowed --onedir `
  --name BlackHoleSimulator `
  --icon assets\blackhole.ico `
  --add-data "web;web" `
  --exclude-module tkinter --exclude-module matplotlib --exclude-module numpy `
  --exclude-module PIL --exclude-module PyQt5 --exclude-module pandas `
  --exclude-module scipy `
  launcher.py

Write-Host "[3/3] copying docs into the portable folder ..."
# (wildcards: this script is ASCII, the doc file name is Chinese)
Copy-Item *.md dist\BlackHoleSimulator\ -Force
Copy-Item *.txt dist\BlackHoleSimulator\ -Force
New-Item -ItemType Directory -Force -Path dist\BlackHoleSimulator\docs | Out-Null
Copy-Item docs\*.webp dist\BlackHoleSimulator\docs\ -Force -ErrorAction SilentlyContinue
Copy-Item docs\*.png  dist\BlackHoleSimulator\docs\ -Force -ErrorAction SilentlyContinue

$size = (Get-ChildItem dist\BlackHoleSimulator -Recurse -File |
         Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("done: dist\BlackHoleSimulator  ({0:N1} MB)" -f $size)
