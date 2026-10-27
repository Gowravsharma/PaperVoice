# install.ps1 - install PaperVoice on Windows in one go.
#
#   One line, in PowerShell:   irm https://raw.githubusercontent.com/Gowravsharma/PaperVoice/main/install.ps1 | iex
#   Or, in a downloaded copy:  powershell -ExecutionPolicy Bypass -File install.ps1
# ---------
# What it does (each step is skipped when already done, so running it again updates the install):
#   1. the PaperVoice code: this folder if the script is run from a copy, else a download to %USERPROFILE%\PaperVoice
#   2. its own tools, INSIDE that folder (tools\): uv (the tested version), the Python it runs on, and Node.js 18+ if
#      the computer has none - so nothing the user has (Python, uv, Node, pip/npm packages, PATH, environment
#      variables) is installed over, changed or even looked at by the packages; no admin rights
#   3. the Python and JavaScript parts into .venv and node_modules (exactly the tested versions: uv.lock,
#      package-lock.json)
#   4. the models (~750 MB, checked against their SHA-256), so the first reading does not wait for them
#   5. the small helper that lets the browser extension start the reader (current user only; its own registry keys
#      and %LOCALAPPDATA%\PaperVoice)
# Then it opens the browser's extensions page for the one step a script cannot do: loading the extension.
# Undo: uninstall.bat, then delete the folder. Nothing else was touched, so nothing else needs undoing.

# -Update: refresh the code of this copy first (git pull, or the repository's zip laid over it; models\ and the
# user's data are kept), then do the same steps - but only if there is something new (-Force: anyway). update.bat
# runs it. -Refreshed is only for the second pass, below.
param([switch]$Update, [switch]$Refreshed, [switch]$Force)

$ErrorActionPreference = "Stop"
# Windows PowerShell 5.1 draws a progress bar for every download and is up to 40x slower for it (a 36 MB file:
# 41 s with the bar, 1 s without; measured) - far worse on a slow computer. Turn it off for the whole script.
$ProgressPreference = "SilentlyContinue"
# older Windows 10 offers only TLS 1.0/1.1 by default; nodejs.org, GitHub and astral.sh need 1.2
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Repo = "Gowravsharma/PaperVoice"                    # the GitHub repository (owner/name) to download from
$UvVersion = "0.13.1"                                # the uv the lock file and the tests were made with
# The user's own settings must not reach into the install (a Python venv, uv or pip variables they use elsewhere)
foreach ($v in "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "PIP_INDEX_URL") {
  Remove-Item "Env:$v" -ErrorAction SilentlyContinue # (this process only; the user's variables stay as they are)
}
$Target = Join-Path $env:USERPROFILE "PaperVoice"  # where a download goes

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host ""; Write-Host "PaperVoice was not installed: $text" -ForegroundColor Red; throw $text }
function Run($exe, [string[]]$arguments) {
  & $exe @arguments
  if ($LASTEXITCODE -ne 0) { Fail "'$exe $($arguments -join ' ')' failed (exit code $LASTEXITCODE)" }
}
function Has($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }
function Download($url, $out) {
  # curl.exe (in Windows 10 and later) shows a progress bar of its own, retries, and gives up on a stalled
  # connection (under 1 KB/s for 60 s); without it, Invoke-WebRequest with a time limit
  $name = Split-Path $url -Leaf
  Write-Host "Downloading $name ..."
  $t = Get-Date
  $curl = Join-Path $env:SystemRoot "System32\curl.exe"
  if (Test-Path $curl) {
    & $curl -L --fail --retry 3 --retry-delay 3 --connect-timeout 20 --speed-limit 1000 --speed-time 60 --progress-bar -o $out $url
    if ($LASTEXITCODE -ne 0) { Fail "the download of $url failed (curl exit code $LASTEXITCODE). Check the internet connection and run this again." }
  } else {
    try { Invoke-WebRequest $url -OutFile $out -UseBasicParsing -TimeoutSec 900 }
    catch { Fail "the download of $url failed: $($_.Exception.Message). Check the internet connection and run this again." }
  }
  Write-Host ("  done in {0} s" -f [int]((Get-Date) - $t).TotalSeconds)
}
function Get-Version($app) {
  try { return (Get-Content (Join-Path $app "version.json") -Raw | ConvertFrom-Json).version } catch { return "unknown" }
}
function Stop-Engine {
  # a running engine holds files of the install (python.exe in .venv) that an update replaces
  try { Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8765/api/shutdown" -TimeoutSec 3 | Out-Null; Start-Sleep -Seconds 2 } catch { }
}
function Newer-Available($app) {
  # $true / $false: is there a newer version than this copy? A git copy compares commits (so a fix pushed without a
  # version bump counts); a zip copy compares version.json with GitHub's. PAPERVOICE_VERSION_URL replaces the
  # address (a mirror, or testing). $null: could not tell.
  try {
    if (Test-Path (Join-Path $app ".git")) {
      if (-not (Has "git")) { return $null }
      & git -C $app fetch --quiet 2>$null
      if ($LASTEXITCODE -ne 0) { return $null }
      $behind = & git -C $app rev-list --count "HEAD..@{u}" 2>$null
      if ($LASTEXITCODE -ne 0) { return $null } # no upstream set: let the pull decide
      return ([int]$behind -gt 0)
    }
    $url = if ($env:PAPERVOICE_VERSION_URL) { $env:PAPERVOICE_VERSION_URL } else { "https://raw.githubusercontent.com/$Repo/main/version.json" }
    $latest = (Invoke-RestMethod $url -TimeoutSec 30).version
    return ([version]$latest -gt [version](Get-Version $app))
  } catch { return $null }
}
function Overlay-Zip($app) {
  # the repository's files laid over the folder: new and changed files replace the old ones; models\, out\, tools\,
  # .venv and node_modules are not in the zip, so they stay. (A file deleted upstream stays as a harmless leftover.)
  # PAPERVOICE_ZIP_URL replaces the address - for a mirror, or for testing.
  $url = if ($env:PAPERVOICE_ZIP_URL) { $env:PAPERVOICE_ZIP_URL } else { "https://github.com/$Repo/archive/refs/heads/main.zip" }
  $zip = Join-Path $env:TEMP "papervoice.zip"
  $tmp = Join-Path $env:TEMP "papervoice-unpacked"
  Download $url $zip
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

# 1. the code ------------------------------------------------------------------------------------------------------
Step "Getting PaperVoice"
$here = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
if ($here -and (Test-Path (Join-Path $here "pyproject.toml"))) {
  $App = $here
  if ($Update -and -not $Refreshed) {
    $now = Get-Version $App
    Step "Looking for a newer version (this copy: $now)"
    $installed = Test-Path (Join-Path $App ".venv\Scripts\python.exe")
    $newer = Newer-Available $App
    if ($installed -and -not $Force) {
      if ($newer -eq $false) {
        Write-Host "PaperVoice $now is already the newest version. Nothing to update." -ForegroundColor Green
        Write-Host "(To reinstall it anyway: update.bat -Force)"
        exit 0
      }
      if ($null -eq $newer) { Write-Host "Could not check (no internet?): updating anyway." }
    }
    if ($newer) { Write-Host "A newer version is available." }
    Step "Updating PaperVoice"
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

# 2. tools, all inside the folder ------------------------------------------------------------------------------------
# uv would otherwise use shared places: its cache and its downloaded Pythons under the user's profile. These
# variables are for this script only. (uv's own installer is not used: even with a target folder it writes its
# "install receipt" into %LOCALAPPDATA%\uv, overwriting the one of a uv the user installed themselves - found by
# comparing the profile before and after an install. The release zip below writes nothing outside the folder.)
$tools = Join-Path $App "tools"
$env:UV_CACHE_DIR = Join-Path $tools "uv-cache"          # removed again at the end (the venv does not need it)
$env:UV_PYTHON_INSTALL_DIR = Join-Path $tools "python"   # PaperVoice's own Python, not the user's
$env:UV_PYTHON_PREFERENCE = "only-managed"               # ... and never one of the user's Pythons

Step "Getting uv $UvVersion (private)"
$uv = Join-Path $tools "uv\uv.exe"
$haveUv = (Test-Path $uv) -and ((& $uv --version) -match [regex]::Escape($UvVersion))
if (-not $haveUv) {
  $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
  $asset = "uv-$arch-pc-windows-msvc.zip"
  $base = "https://github.com/astral-sh/uv/releases/download/$UvVersion"
  $zip = Join-Path $env:TEMP $asset
  Download "$base/$asset" $zip
  $want = ((Invoke-WebRequest "$base/$asset.sha256" -UseBasicParsing -TimeoutSec 60).Content -split "\s+")[0].ToLower()
  $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
  if (-not $want -or $got -ne $want) { Remove-Item $zip -Force; Fail "the uv download did not match its published checksum: run this again." }
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $tmp = Join-Path $env:TEMP "papervoice-uv"
  if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp)
  New-Item -ItemType Directory -Force (Join-Path $tools "uv") | Out-Null
  Copy-Item (Join-Path $tmp "*.exe") (Join-Path $tools "uv") -Force
  Remove-Item $tmp, $zip -Recurse -Force
  if (-not (Test-Path $uv)) { Fail "uv could not be unpacked. Run this again." }
}
Write-Host "uv: $(& $uv --version)  ($uv)"

# Node.js: the one installed (18 or newer) is only run, never changed; without one, a private copy in tools\node -
# no winget, no admin (the reader finds it there: script.py node_exe)
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
  $index = Invoke-RestMethod "https://nodejs.org/dist/index.json" -TimeoutSec 60
  $release = $index | Where-Object { $_.lts -and ($_.files -contains "win-$arch-zip") } | Select-Object -First 1
  if (-not $release) { Fail "could not find a Node.js release for this computer: install the LTS version from https://nodejs.org, then run this again." }
  $name = "node-$($release.version)-win-$arch"
  $zip = Join-Path $env:TEMP "$name.zip"
  Download "https://nodejs.org/dist/$($release.version)/$name.zip" $zip
  $sums = (Invoke-WebRequest "https://nodejs.org/dist/$($release.version)/SHASUMS256.txt" -UseBasicParsing -TimeoutSec 60).Content
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
# a copy that was only downloaded or cloned and never installed is installed, not updated (what the last step says)
$asUpdate = $Update -and (Test-Path (Join-Path $App ".venv\Scripts\python.exe"))
Stop-Engine
Step "Installing the Python parts (uv sync)"
Run $uv @("sync", "--locked")
# before the 750 MB of models: can this computer load the native libraries at all? (a report, not a raw crash)
Step "Checking that this computer can run the models"
Run $uv @("run", "--no-sync", "papervoice-doctor")
Step "Installing the maths speech engine and the PDF viewer (npm ci)"
# npm.cmd, not npm: "npm" can resolve to npm.ps1, which PowerShell's default policy does not let run
# (npm's cache is private too, so the user's is not filled with ours)
$npmCache = Join-Path $tools "npm-cache"
Run "npm.cmd" @("ci", "--prefix", "speech", "--no-audit", "--no-fund", "--cache", $npmCache)
Run "npm.cmd" @("ci", "--prefix", "viewer", "--no-audit", "--no-fund", "--cache", $npmCache)

# 4. models --------------------------------------------------------------------------------------------------------
Step "Downloading the models (once, ~750 MB; each file checked)"
Run $uv @("run", "--no-sync", "papervoice-models")

# 5. the browser helper --------------------------------------------------------------------------------------------
Step "Letting the browser extension start the reader"
Run $uv @("run", "--no-sync", "papervoice-setup")

# the caches were only for installing: the venv and node_modules have their own copies of the files
foreach ($c in $env:UV_CACHE_DIR, $npmCache) { try { Remove-Item $c -Recurse -Force -ErrorAction Stop } catch { } }

# the extension -------------------------------------------------------------------------------------------------------
$ext = Join-Path $App "extension"
if ($asUpdate) {
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
Write-Host "PaperVoice $(Get-Version $App) is $(if ($asUpdate) { 'updated' } else { 'installed' }) in $App" -ForegroundColor Green
