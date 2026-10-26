# install.ps1 - install PaperVoice on Windows in one go.
#
#   One line, in PowerShell:   irm https://raw.githubusercontent.com/Gowravsharma/PaperVoice/main/install.ps1 | iex
#   Or, in a downloaded copy:  powershell -ExecutionPolicy Bypass -File install.ps1
#
# What it does (each step is skipped when already done, so running it again updates the install):
#   1. uv (Python manager, installs Python by itself) and Node.js 18+, if missing - from their official sources
#   2. the PaperVoice code: this folder if the script is run from a copy, else a download to %USERPROFILE%\PaperVoice
#   3. its Python and JavaScript parts (exactly the tested versions: uv.lock, package-lock.json)
#   4. the models (~750 MB, checked against their SHA-256), so the first reading does not wait for them
#   5. the small helper that lets the browser extension start the reader (current user only, no admin)
# Then it opens the browser's extensions page for the one step a script cannot do: loading the extension.
# Undo: uv run papervoice-setup --remove, then delete the folder.

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

if ($env:OS -ne "Windows_NT") { Fail "this installer is for Windows. On macOS/Linux follow the README's manual steps." }

# 1. tools ---------------------------------------------------------------------------------------------------------
Step "Checking uv and Node.js"
if (-not (Has "uv")) {
  Write-Host "Installing uv (https://docs.astral.sh/uv/)..."
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  Refresh-Path
  if (-not (Has "uv")) { Fail "uv was installed but is not found. Open a new PowerShell window and run this again." }
}
Write-Host "uv: $(uv --version)"

$nodeOk = $false
if (Has "node") { $nodeOk = [int]((node --version).TrimStart("v").Split(".")[0]) -ge 18 }
if (-not $nodeOk) {
  if (-not (Has "winget")) { Fail "Node.js 18 or newer is needed: install the LTS version from https://nodejs.org, then run this again." }
  Write-Host "Installing Node.js LTS with winget..."
  Run "winget" @("install", "--id", "OpenJS.NodeJS.LTS", "-e", "--silent", "--accept-package-agreements", "--accept-source-agreements")
  Refresh-Path
  if (-not (Has "node")) { Fail "Node.js was installed but is not found. Open a new PowerShell window and run this again." }
}
Write-Host "Node.js: $(node --version)"

# 2. the code ------------------------------------------------------------------------------------------------------
Step "Getting PaperVoice"
$here = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
if ($here -and (Test-Path (Join-Path $here "pyproject.toml"))) {
  $App = $here
  Write-Host "Using this copy: $App"
} else {
  $App = $Target
  if (Test-Path (Join-Path $App ".git")) {
    Write-Host "Updating $App ..."
    Run "git" @("-C", $App, "pull", "--ff-only")
  } elseif (Has "git") {
    if (Test-Path $App) { Fail "$App exists and is not a PaperVoice download: move it away and run this again." }
    Run "git" @("clone", "--depth", "1", "https://github.com/$Repo.git", $App)
  } else { # no git: the same files as a zip
    $zip = Join-Path $env:TEMP "papervoice.zip"
    $unpacked = Join-Path $env:TEMP "papervoice-unpacked"
    Invoke-WebRequest "https://github.com/$Repo/archive/refs/heads/main.zip" -OutFile $zip -UseBasicParsing
    if (Test-Path $unpacked) { Remove-Item $unpacked -Recurse -Force }
    Expand-Archive $zip $unpacked
    $inner = Get-ChildItem $unpacked -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Force $App | Out-Null
    Copy-Item (Join-Path $inner.FullName "*") $App -Recurse -Force # an update keeps models\ and out\
    Remove-Item $zip, $unpacked -Recurse -Force
  }
}
Set-Location $App

# 3. its parts -----------------------------------------------------------------------------------------------------
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

# the extension: the one step left -----------------------------------------------------------------------------------
$ext = Join-Path $App "extension"
if (-not $env:PAPERVOICE_NO_BROWSER) { Set-Clipboard -Value $ext }
Step "Last step: add the extension (once)"
Write-Host "  1. In the page that opens, switch on 'Developer mode'."
Write-Host "  2. Click 'Load unpacked' and choose this folder (its path is on your clipboard - paste it):"
Write-Host "       $ext" -ForegroundColor Yellow
Write-Host "  3. Open any PDF in the browser and click the PaperVoice icon -> 'Read this PDF aloud'."
Write-Host ""
$edge = Get-Command msedge -ErrorAction SilentlyContinue
$edgeExe = Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"
$chromeExe = Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"
if ($env:PAPERVOICE_NO_BROWSER) { } # (tests: no browser tab, clipboard untouched)
elseif (Test-Path $chromeExe) { Start-Process $chromeExe "chrome://extensions" }
elseif ($edge -or (Test-Path $edgeExe)) { Start-Process $(if ($edge) { $edge.Source } else { $edgeExe }) "edge://extensions" }
Write-Host "PaperVoice is installed in $App" -ForegroundColor Green
