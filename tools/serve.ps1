# tools/serve.ps1 — tiny static HTTP server for local testing (no Node needed).
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\serve.ps1 -Port 8080
# Serves the project folder at http://localhost:8080/ . Ctrl+C to stop.
param([int]$Port = 8080)
$root = Split-Path -Parent $PSScriptRoot
$mime = @{ '.html'='text/html; charset=utf-8'; '.js'='text/javascript; charset=utf-8'; '.css'='text/css; charset=utf-8'; '.json'='application/json'; '.png'='image/png'; '.jpg'='image/jpeg'; '.svg'='image/svg+xml'; '.md'='text/plain; charset=utf-8'; '.ico'='image/x-icon'; '.txt'='text/plain; charset=utf-8' }
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:$Port/")
try { $listener.Start() } catch { Write-Host "Could not listen on port $Port : $($_.Exception.Message)"; exit 1 }
Write-Host "Serving $root at http://localhost:$Port/  (Ctrl+C to stop)"
while ($listener.IsListening) {
  try {
    $ctx = $listener.GetContext()
    $path = [System.Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath)
    if ($path -eq '/') { $path = '/index.html' }
    $file = Join-Path $root ($path -replace '/', '\')
    $res = $ctx.Response
    $res.Headers.Add('Cache-Control', 'no-store')
    if ((Test-Path $file -PathType Leaf) -and ($file.StartsWith($root))) {
      $bytes = [System.IO.File]::ReadAllBytes($file)
      $ext = [System.IO.Path]::GetExtension($file).ToLower()
      $res.ContentType = if ($mime[$ext]) { $mime[$ext] } else { 'application/octet-stream' }
      $res.ContentLength64 = $bytes.Length
      $res.OutputStream.Write($bytes, 0, $bytes.Length)
    } else {
      $res.StatusCode = 404
      $msg = [System.Text.Encoding]::UTF8.GetBytes("404 $path")
      $res.OutputStream.Write($msg, 0, $msg.Length)
    }
    $res.OutputStream.Close()
  } catch { Write-Host "error: $($_.Exception.Message)" }
}
