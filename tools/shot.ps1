# tools/shot.ps1 — screenshot a lesson step (end state, deterministic) with headless Chrome.
#   .\tools\shot.ps1 p2-01-linear-transformations 3          → tools\shots\p2-01-linear-transformations-3.png
#   .\tools\shot.ps1 <id> <step> -Mode play                  → playground view
#   .\tools\shot.ps1 <id> <step> -Mode check -Check 1        → check mode, second check selected
#   .\tools\shot.ps1 -Url "#/selftest?gallery=1&only=<id>" -H 3000   → arbitrary route (gallery page)
# Prints the png path. Read the png to look at it.
param(
  [string]$Id = "",
  [int]$Step = 0,
  [string]$Mode = "",
  [int]$Check = -1,
  [int]$W = 1280,
  [int]$H = 800,
  [string]$Url = "",
  [string]$Out = "",
  [int]$Budget = 10000,
  [double]$T = -1
)
$root = Split-Path -Parent $PSScriptRoot
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chrome)) { $chrome = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" }
if (-not (Test-Path $chrome)) { Write-Host "No Chrome/Edge found"; exit 2 }
$profile = Join-Path $env:TEMP ("la-headless-" + [guid]::NewGuid().ToString('N').Substring(0, 8))   # unique per run
$base = "file:///" + ($root -replace '\\', '/') + "/index.html"
if ($Url) { $full = $base + $Url }
else {
  if (-not $Id) { Write-Host "usage: shot.ps1 <lesson-id> <step> [-Mode play|check] [-Check n]"; exit 2 }
  $q = "instant=1"
  if ($Mode) { $q += "&mode=$Mode" }
  if ($Check -ge 0) { $q += "&check=$Check" }
  if ($T -ge 0) { $q += "&t=$T" }
  $full = "$base#/lesson/$Id/$Step" + "?" + $q
}
if ($Out -and -not [System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path (Get-Location).Path $Out }   # Chrome resolves relative paths from its own cwd
$shots = Join-Path $root "tools\shots"
if (-not (Test-Path $shots)) { New-Item -ItemType Directory -Path $shots | Out-Null }
if (-not $Out) {
  $name = if ($Id) { "$Id-$Step" + $(if ($Mode) { "-$Mode" } else { "" }) + $(if ($Check -ge 0) { "-c$Check" } else { "" }) } else { "route-" + [guid]::NewGuid().ToString('N').Substring(0, 6) }
  $Out = Join-Path $shots "$name.png"
}
$args = @('--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', "--user-data-dir=$profile", '--allow-file-access-from-files', '--hide-scrollbars', "--window-size=$W,$H", "--virtual-time-budget=$Budget", "--screenshot=$Out", $full)
# Headless Chrome occasionally captures before the first paint (an all-black, tiny PNG); retry up to twice.
for ($try = 0; $try -lt 3; $try++) {
  if (Test-Path $Out) { Remove-Item $Out -Force -ErrorAction SilentlyContinue }
  Start-Process -FilePath $chrome -ArgumentList $args -NoNewWindow -Wait | Out-Null
  if ((Test-Path $Out) -and ((Get-Item $Out).Length -gt 20000)) { break }
  $args = $args | ForEach-Object { if ($_ -like '--virtual-time-budget=*') { "--virtual-time-budget=$($Budget + 4000 * ($try + 1))" } else { $_ } }
}
Remove-Item $profile -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path $Out) { Write-Output $Out } else { Write-Host "screenshot failed"; exit 1 }
