# install.ps1 - install PaperVoice on Windows in one go.
#
#   One line, in PowerShell:   irm https://raw.githubusercontent.com/Gowravsharma/PaperVoice/main/install.ps1 | iex
#   Or, in a downloaded copy:  powershell -ExecutionPolicy Bypass -File install.ps1
# ---------
# What it does (each step is skipped when already done, so running it again updates the install):
#   1. uv (Python manager, installs Python by itself), if missing - from its official installer
#   2. the PaperVoice code: this folder if the script is run from a copy, else a download to %USERPROFILE%\PaperVoice
#   3. Node.js 18+ if missing (a private, checksum-verified copy in tools/node; no admin), then its Python and
#      JavaScript parts (exactly the tested versions: uv.lock, package-lock.json)
#   4. the models (~750 MB, checked against their SHA-256), so the first reading does not wait for them
#   5. the small helper that lets the browser extension start the reader (current user only, no admin)
# Then it opens the browser's extensions page for the one step a script cannot do: loading the extension.
# Undo: uv run papervoice-setup --remove, then delete the folder.

# -Update: refresh the code of this copy first (git pull, or the repository's zip laid over it; models\ and the
# user's data are kept), then do the same steps. update.bat runs it. -Refreshed is only for the second pass, below.
param([switch]$Update, [switch]$Refreshed)

$ErrorActionPreference = "Stop"
$Repo = "Gowravsharma/PaperVoice"                    # the GitHub repository (owner/name) to download from
$Target = Join-Path $env:USERPROFILE "PaperVoice"  # where a download goes

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host ""; Write-Host "PaperVoice was not installed: $text" -ForegroundColor Red; throw $text }
function Run($exe, [string[]]$arguments) {
  & $exe @arguments
  if ($LASTEXITCODE -ne 0) { Fail "'$exe $($arguments -join ' ')' failed (exit code $LASTEXITCODE)" }
}
function Refresh-Path {
  # programs installed a moment ago are on the PATH of new windows only: read it again for this one
  $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
  $user = [Environment]::GetEnvironmentVariable("Path", "User")
  $env:Path = "$machine;$user;$(Join-Path $env:USERPROFILE '.local\bin')"
}
function Has($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }
function Get-Version($app) {
  try { return (Get-Content (Join-Path $app "version.json") -Raw | ConvertFrom-Json).version } catch { return "unknown" }
}
function Stop-Engine {
  # a running engine holds files of the install (python.exe in .venv) that an update replaces
  try { Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8765/api/shutdown" -TimeoutSec 3 | Out-Null; Start-Sleep -Seconds 2 } catch { }
}
function Overlay-Zip($app) {
  # the repository's files laid over the folder: new and changed files replace the old ones; models\, out\, tools\,
  # .venv and node_modules are not in the zip, so they stay. (A file deleted upstream stays as a harmless leftover.)
  # PAPERVOICE_ZIP_URL replaces the address - for a mirror, or for testing.
  $url = if ($env:PAPERVOICE_ZIP_URL) { $env:PAPERVOICE_ZIP_URL } else { "https://github.com/$Repo/archive/refs/heads/main.zip" }
  $zip = Join-Path $env:TEMP "papervoice.zip"
  $tmp = Join-Path $env:TEMP "papervoice-unpacked"
  Invoke-WebRequest $url -OutFile $zip -UseBasicParsing
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp)
  $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1 # GitHub's zip has one folder, PaperVoice-main
  if (-not $inner -or -not (Test-Path (Join-Path $inner.FullName "pyproject.toml"))) { Fail "the download is not a PaperVoice zip." }
  New-Item -ItemType Directory -Force $app | Out-Null
  Copy-Item (Join-Path $inner.FullName "*") $app -Recurse -Force
  Remove-Item $zip, $tmp -Recurse -Force
}

if ($env:OS -ne "Windows_NT") { Fail "this installer is for Windows. On macOS/Linux follow the README's manual steps." }

# 1. tools ---------------------------------------------------------------------------------------------------------
Step "Checking uv"
if (-not (Has "uv")) {
  Write-Host "Installing uv (https://docs.astral.sh/uv/)..."
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  Refresh-Path
  if (-not (Has "uv")) { Fail "uv was installed but is not found. Open a new PowerShell window and run this again." }
}
Write-Host "uv: $(uv --version)"

Write-Host "(Node.js is checked after the download, below)"

# 2. the code ------------------------------------------------------------------------------------------------------
Step "Getting PaperVoice"
$here = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
if ($here -and (Test-Path (Join-Path $here "pyproject.toml"))) {
  $App = $here
  if ($Update -and -not $Refreshed) {
    Step "Updating PaperVoice (now $(Get-Version $App))"
    Stop-Engine
    if (Test-Path (Join-Path $App ".git")) {
      if (-not (Has "git")) { Fail "this folder came from git but git is not installed." }
      Run "git" @("-C", $App, "pull", "--ff-only")
    } else {
      Overlay-Zip $App
    }
    # the new install.ps1 may have steps this one does not know: carry on with it
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $App "install.ps1") -Update -Refreshed
    exit $LASTEXITCODE
  }
  Write-Host "Using this copy: $App"
} else {
  $App = $Target
  if (Test-Path (Join-Path $App "pyproject.toml")) { $Update = $true } # run again over an existing install
  if ($Update) { Stop-Engine }
  if (Test-Path (Join-Path $App ".git")) {
    Write-Host "Updating $App ..."
    Run "git" @("-C", $App, "pull", "--ff-only")
  } elseif (Has "git") {
    if (Test-Path $App) { Fail "$App exists and is not a PaperVoice download: move it away and run this again." }
    Run "git" @("clone", "--depth", "1", "https://github.com/$Repo.git", $App)
  } else { # no git: the same files as a zip
    Overlay-Zip $App
  }
}
Set-Location $App

# Node.js: the one installed (18 or newer), else a private copy in tools\node - no winget, no admin, nothing else
# on the computer is touched (the reader finds it there: script.py node_exe)
Step "Checking Node.js"
$nodeOk = $false
if (Has "node") { $nodeOk = [int]((node --version).TrimStart("v").Split(".")[0]) -ge 18 }
$privateNode = Join-Path $App "tools\node"
if (-not $nodeOk -and (Test-Path (Join-Path $privateNode "node.exe"))) {
  $env:Path = "$privateNode;$env:Path"
  $nodeOk = $true
}
if (-not $nodeOk) {
  Write-Host "Node.js 18+ is not installed: downloading a private copy (about 30 MB) from nodejs.org..."
  $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
  $index = Invoke-RestMethod "https://nodejs.org/dist/index.json"
  $release = $index | Where-Object { $_.lts -and ($_.files -contains "win-$arch-zip") } | Select-Object -First 1
  if (-not $release) { Fail "could not find a Node.js release for this computer: install the LTS version from https://nodejs.org, then run this again." }
  $name = "node-$($release.version)-win-$arch"
  $zip = Join-Path $env:TEMP "$name.zip"
  Invoke-WebRequest "https://nodejs.org/dist/$($release.version)/$name.zip" -OutFile $zip -UseBasicParsing
  $sums = (Invoke-WebRequest "https://nodejs.org/dist/$($release.version)/SHASUMS256.txt" -UseBasicParsing).Content
  $want = ($sums -split "`n" | Where-Object { $_ -match "  $name\.zip\s*$" } | Select-Object -First 1).Split(" ")[0]
  $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
  if (-not $want -or $got -ne $want) { Remove-Item $zip -Force; Fail "the Node.js download did not match its checksum: run this again." }
  # unzipped in a short temp folder and then moved (a rename): Node's npm has very deep folders, and in a long
  # install path Expand-Archive stops at Windows' 260-character limit
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $tmp = Join-Path $env:TEMP "papervoice-node"
  if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp)
  New-Item -ItemType Directory -Force (Join-Path $App "tools") | Out-Null
  if (Test-Path $privateNode) { Remove-Item $privateNode -Recurse -Force }
  Move-Item (Join-Path $tmp $name) $privateNode
  Remove-Item $tmp, $zip -Recurse -Force
  $env:Path = "$privateNode;$env:Path"
}
Write-Host "Node.js: $(node --version)"

# 3. its parts -----------------------------------------------------------------------------------------------------
Stop-Engine
Step "Installing the Python parts (uv sync)"
Run "uv" @("sync", "--locked")
Step "Installing the maths speech engine and the PDF viewer (npm ci)"
# npm.cmd, not npm: "npm" can resolve to npm.ps1, which PowerShell's default policy does not let run
Run "npm.cmd" @("ci", "--prefix", "speech", "--no-audit", "--no-fund")
Run "npm.cmd" @("ci", "--prefix", "viewer", "--no-audit", "--no-fund")

# 4. models --------------------------------------------------------------------------------------------------------
Step "Downloading the models (once, ~750 MB; each file checked)"
Run "uv" @("run", "--no-sync", "papervoice-models")

# 5. the browser helper --------------------------------------------------------------------------------------------
Step "Letting the browser extension start the reader"
Run "uv" @("run", "--no-sync", "papervoice-setup")

# the extension -------------------------------------------------------------------------------------------------------
$ext = Join-Path $App "extension"
if ($Update) {
  Step "Last step: reload the extension"
  Write-Host "  On the extensions page that opens, click the reload icon on the PaperVoice card."
  Write-Host "  (Then click the PaperVoice icon on a PDF as usual: the reading engine starts with the new code.)"
} else {
  if (-not $env:PAPERVOICE_NO_BROWSER) { Set-Clipboard -Value $ext }
  Step "Last step: add the extension (once)"
  Write-Host "  1. In the page that opens, switch on 'Developer mode'."
  Write-Host "  2. Click 'Load unpacked' and choose this folder (its path is on your clipboard - paste it):"
  Write-Host "       $ext" -ForegroundColor Yellow
  Write-Host "  3. Open any PDF in the browser and click the PaperVoice icon -> 'Read this PDF aloud'."
}
Write-Host ""
$edge = Get-Command msedge -ErrorAction SilentlyContinue
$edgeExe = Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"
$chromeExe = Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"
if ($env:PAPERVOICE_NO_BROWSER) { } # (tests: no browser tab, clipboard untouched)
elseif (Test-Path $chromeExe) { Start-Process $chromeExe "chrome://extensions" }
elseif ($edge -or (Test-Path $edgeExe)) { Start-Process $(if ($edge) { $edge.Source } else { $edgeExe }) "edge://extensions" }
Write-Host "PaperVoice $(Get-Version $App) is $(if ($Update) { 'updated' } else { 'installed' }) in $App" -ForegroundColor Green
