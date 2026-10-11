# tools/push.ps1 — first push of this project to GitHub (needs Git for Windows).
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\push.ps1 -Url https://github.com/<user>/linear-algebra-for-ml.git
# Safe to re-run: it reuses an existing repo and remote, commits any changes, and pushes the main branch.
param(
  [Parameter(Mandatory = $true)][string]$Url,
  [string]$Message = "Visual linear algebra for machine learning: 57 interactive 2D/3D lessons",
  [string]$Name = "",
  [string]$Email = ""
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$git = (Get-Command git -ErrorAction SilentlyContinue)
if (-not $git) {
  foreach ($p in "$env:ProgramFiles\Git\cmd\git.exe", "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe") { if (Test-Path $p) { $git = $p } }
  if (-not $git) { Write-Host "Git is not installed. Install Git for Windows from https://git-scm.com/download/win and re-run."; exit 2 }
} else { $git = $git.Source }

if (-not (Test-Path (Join-Path $root '.git'))) { & $git init -b main | Out-Host }

# commit identity: use the global one if set, otherwise the values passed in
$haveName = (& $git config user.name) ; $haveEmail = (& $git config user.email)
if (-not $haveName) { if (-not $Name) { Write-Host "No git user.name set. Re-run with -Name 'Your Name' -Email you@example.com"; exit 2 }; & $git config user.name $Name }
if (-not $haveEmail) { if (-not $Email) { Write-Host "No git user.email set. Re-run with -Email you@example.com"; exit 2 }; & $git config user.email $Email }

& $git add -A
$pending = (& $git status --porcelain)
if ($pending) { & $git commit -m $Message | Out-Host } else { Write-Host "Nothing new to commit." }

$remotes = (& $git remote)
if ($remotes -contains 'origin') { & $git remote set-url origin $Url } else { & $git remote add origin $Url }

& $git branch -M main
Write-Host "Pushing to $Url (a GitHub sign-in window may open the first time)..."
& $git push -u origin main
if ($LASTEXITCODE -eq 0) { Write-Host "Done: $($Url -replace '\.git$','')" } else { Write-Host "Push failed (exit $LASTEXITCODE). Check that the repository exists and that you are signed in."; exit 1 }
