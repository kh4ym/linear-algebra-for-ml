# tools/check.ps1 — run the app's self-test page in headless Chrome and print a PASS/FAIL report.
#   .\tools\check.ps1                 all lessons
#   .\tools\check.ps1 p2-01-linear-transformations     one lesson
#   .\tools\check.ps1 -Json           also print the raw JSON
# Exit code 0 = everything passed, 1 = failures, 2 = could not run.
param(
  [string]$Id = "",
  [switch]$Json,
  [int]$Budget = 20000
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chrome)) { $chrome = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" }
if (-not (Test-Path $chrome)) { Write-Host "No Chrome/Edge found"; exit 2 }
$run = [guid]::NewGuid().ToString('N').Substring(0, 8)
$profile = Join-Path $env:TEMP "la-headless-$run"      # unique per run so parallel runs never collide
$tmp = Join-Path $env:TEMP ("la-selftest-" + $run + ".html")
$url = "file:///" + ($root -replace '\\', '/') + "/index.html#/selftest"
if ($Id) { $url += "?only=$Id" }

$args = @('--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', "--user-data-dir=$profile", '--allow-file-access-from-files', "--virtual-time-budget=$Budget", '--dump-dom', $url)
$p = Start-Process -FilePath $chrome -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmp
if (-not (Test-Path $tmp)) { Write-Host "Chrome produced no output"; exit 2 }
$html = Get-Content $tmp -Raw -Encoding UTF8
Remove-Item $tmp -ErrorAction SilentlyContinue
Remove-Item $profile -Recurse -Force -ErrorAction SilentlyContinue

$m = [regex]::Match($html, '<pre class="hidden" id="selftest-json">(.*?)</pre>', 'Singleline')
if (-not $m.Success) {
  Write-Host "Self-test did not finish. Raise -Budget, or look for a boot error:"
  $err = [regex]::Match($html, '<div id="banner"[^>]*>(.*?)</div>', 'Singleline')
  if ($err.Success) { Write-Host ([System.Net.WebUtility]::HtmlDecode(($err.Groups[1].Value -replace '<[^>]+>', ' '))) }
  $boot = [regex]::Match($html, '<div id="boot-status"[^>]*>(.*?)</div>', 'Singleline')
  if ($boot.Success) { Write-Host ("boot: " + $boot.Groups[1].Value) }
  exit 2
}
$jsonText = [System.Net.WebUtility]::HtmlDecode($m.Groups[1].Value)
if ($Json) { Write-Host $jsonText }
$r = $jsonText | ConvertFrom-Json

$fail = 0
foreach ($L in $r.lessons) {
  $status = if ($L.pass) { 'PASS' } else { 'FAIL'; $fail++ }
  $stepsOk = @($L.steps | Where-Object { $_.ok }).Count
  $checksOk = @($L.checks | Where-Object { $_.ok }).Count
  Write-Host ("{0,-5} {1}  steps {2}/{3}  checks {4}/{5}  spec-violations {6}  {7}ms  records {8}" -f $status, $L.id, $stepsOk, $L.steps.Count, $checksOk, $L.checks.Count, $L.violations.Count, $L.ms, $L.records)
  foreach ($s in $L.steps) { if (-not $s.ok) { Write-Host ("        step {0}: {1}" -f ($s.k + 1), $s.error) } }
  foreach ($c in $L.checks) { if (-not $c.ok) { Write-Host ("        check {0}: {1}" -f ($c.i + 1), $c.error) } elseif ($c.note) { Write-Host ("        check {0} note: {1}" -f ($c.i + 1), $c.note) } }
  if ($L.playground -and -not $L.playground.ok) { Write-Host ("        playground: " + $L.playground.error) }
  foreach ($v in $L.violations) { Write-Host ("        spec: " + $v) }
  foreach ($s in $L.steps) { if ($s.words -and ($s.words -lt 20 -or $s.words -gt 70)) { Write-Host ("        words: step {0} has {1} words" -f ($s.k + 1), $s.words) } }
}
$numFail = @($r.num | Where-Object { -not $_.pass })
foreach ($n in $numFail) { Write-Host ("NUM FAIL  {0}: {1}" -f $n.name, $n.detail) }
foreach ($e in $r.loadErrors) { Write-Host ("LOAD      " + $e) }
foreach ($u in $r.unlisted) { Write-Host ("UNLISTED  " + $u) }
Write-Host ("")
Write-Host ("SELFTEST {0}: lessons {1}/{2} pass, numeric {3}/{4} pass, planned-not-built {5}" -f $r.status.ToUpper(), (@($r.lessons | Where-Object { $_.pass }).Count), $r.lessons.Count, (@($r.num | Where-Object { $_.pass }).Count), $r.num.Count, $r.planned.Count)
if ($r.status -eq 'pass') { exit 0 } else { exit 1 }
