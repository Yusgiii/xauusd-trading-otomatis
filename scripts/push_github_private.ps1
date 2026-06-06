# Push proyek ke GitHub PRIVATE (hanya Anda yang bisa lihat).
# Jalankan SETELAH: gh auth login
#
# PowerShell:
#   cd "D:\trading\trading otomatis 5"
#   .\scripts\push_github_private.ps1

$ErrorActionPreference = "Stop"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) {
    Write-Host "GitHub CLI belum terpasang. Install: winget install GitHub.cli"
    exit 1
}

& $gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Belum login GitHub. Jalankan: gh auth login"
    exit 1
}

$repoName = "xauusd-trading-otomatis"
Write-Host "Membuat repo PRIVATE: $repoName ..."

& $gh repo create $repoName --private --source=. --remote=origin --push --description "XAUUSD M15 trading automation (private)"

if ($LASTEXITCODE -eq 0) {
    $url = & $gh repo view --json url -q .url
    Write-Host ""
    Write-Host "Selesai! Repo private: $url"
    Write-Host "Hanya akun GitHub Anda yang punya akses."
} else {
    Write-Host "Gagal. Jika repo sudah ada, coba:"
    Write-Host "  git remote add origin https://github.com/USERNAME/$repoName.git"
    Write-Host "  git push -u origin master"
}
