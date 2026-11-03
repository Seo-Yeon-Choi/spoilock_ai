# SPOILOCK Python 폴더 정리 (C:\spoilock_cleaned 에서 실행)
#   1단계: 아래 파일이 다른 코드에서 쓰이는지 확인만 함 (아무것도 바꾸지 않음)
#   2단계: -Apply 를 붙이면 archive\cleanup-20261103 으로 '이동' (삭제 아님)
#   powershell -ExecutionPolicy Bypass -File .\cleanup_check.ps1
#   powershell -ExecutionPolicy Bypass -File .\cleanup_check.ps1 -Apply
param([switch]$Apply)
Set-Location $PSScriptRoot

$candidates = @("spoiler_server_base.py", "video_srt_experiment.py", "env.example", "gitignore")
$code = Get-ChildItem -Path . -Filter *.py -File | Where-Object { $candidates -notcontains $_.Name }

Write-Host "`n[1] 파이썬 파일이 다른 코드에서 import 되는지" -ForegroundColor Cyan
foreach ($name in @("spoiler_server_base", "video_srt_experiment")) {
    $hits = $code | Select-String -Pattern "\b$name\b" -List
    if ($hits) { Write-Host "  $name.py : 사용 중 -> 지우면 안 됨" -ForegroundColor Red; $hits | ForEach-Object { "     " + $_.Filename + ":" + $_.LineNumber } }
    else       { Write-Host "  $name.py : 어디서도 쓰지 않음" -ForegroundColor Green }
}

Write-Host "`n[2] 점(.) 없는 중복 파일 비교" -ForegroundColor Cyan
foreach ($pair in @(@("env.example", ".env.example"), @("gitignore", ".gitignore"))) {
    $a, $b = $pair
    if (-not (Test-Path $a)) { continue }
    $same = (Get-FileHash $a).Hash -eq (Get-FileHash $b).Hash
    if ($same) { Write-Host "  $a = $b (내용 같음)" -ForegroundColor Green }
    else       { Write-Host "  $a 와 $b 내용 다름 -> 아래 차이 확인" -ForegroundColor Yellow; Compare-Object (Get-Content $b) (Get-Content $a) | Format-Table -AutoSize }
}

if (-not $Apply) { Write-Host "`n확인만 했습니다. 이동하려면 -Apply 를 붙여 다시 실행하세요.`n"; exit }

$dest = "archive\cleanup-20261103"
New-Item -ItemType Directory -Force $dest | Out-Null
foreach ($f in $candidates + @("experiments")) {
    if (Test-Path $f) { Move-Item $f $dest -Force; Write-Host "  이동: $f" }
}
if (Test-Path "__pycache__") { Remove-Item "__pycache__" -Recurse -Force; Write-Host "  삭제: __pycache__ (자동 생성 캐시)" }
Write-Host "`n완료. 서버를 켜서 정상 동작을 확인한 뒤 archive\cleanup-20261103 을 지우면 됩니다.`n"
