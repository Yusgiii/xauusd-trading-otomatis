# Pasang shortcut Startup: MetaTrader 5 + Bot XAUUSD H1 Stage 9 (on-demand).
# Jalankan sekali (PowerShell biasa cukup):
#   cd "D:\trading\trading otomatis 5"
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_autostart.ps1
#
# Hapus:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_autostart.ps1 -Uninstall

param(
    [switch]$Uninstall,
    [string]$Mt5Path = ""
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Startup = [Environment]::GetFolderPath("Startup")
$Shell = New-Object -ComObject WScript.Shell

$LnkMt5 = Join-Path $Startup "XAUUSD_MetaTrader5.lnk"
$LnkBot = Join-Path $Startup "XAUUSD_H1_Stage9_Bot.lnk"
$VbsBot = Join-Path $ProjectDir "scripts\run_stage9_service_hidden.vbs"

function Find-Mt5Terminal {
    param([string]$Manual)
    if ($Manual -and (Test-Path $Manual)) { return (Resolve-Path $Manual).Path }

    $candidates = @(
        "${env:ProgramFiles}\MetaTrader 5\terminal64.exe",
        "${env:ProgramFiles(x86)}\MetaTrader 5\terminal64.exe",
        "$env:APPDATA\MetaQuotes\Terminal\*\terminal64.exe"
    )
    foreach ($p in $candidates) {
        $found = Get-Item $p -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

function New-Shortcut {
    param(
        [string]$LinkPath,
        [string]$Target,
        [string]$Arguments = "",
        [string]$WorkingDirectory = "",
        [string]$Description = ""
    )
    $sc = $Shell.CreateShortcut($LinkPath)
    $sc.TargetPath = $Target
    if ($Arguments) { $sc.Arguments = $Arguments }
    if ($WorkingDirectory) { $sc.WorkingDirectory = $WorkingDirectory }
    if ($Description) { $sc.Description = $Description }
    $sc.Save()
}

if ($Uninstall) {
    foreach ($p in @($LnkMt5, $LnkBot)) {
        if (Test-Path $p) {
            Remove-Item $p -Force
            Write-Host "[HAPUS] $p"
        }
    }
    Write-Host "Autostart dihapus dari: $Startup"
    exit 0
}

if (-not (Test-Path $VbsBot)) {
    throw "Tidak ditemukan: $VbsBot"
}

# --- Bot Stage 9 ---
New-Shortcut `
    -LinkPath $LnkBot `
    -Target "wscript.exe" `
    -Arguments "`"$VbsBot`"" `
    -WorkingDirectory $ProjectDir `
    -Description "XAUUSD H1 Telegram bot on-demand (/analisa)"
Write-Host "[OK] Bot -> $LnkBot"

# --- MetaTrader 5 ---
$mt5 = Find-Mt5Terminal -Manual $Mt5Path
if ($mt5) {
    New-Shortcut `
        -LinkPath $LnkMt5 `
        -Target $mt5 `
        -WorkingDirectory (Split-Path $mt5 -Parent) `
        -Description "MetaTrader 5 auto-start"
    Write-Host "[OK] MT5  -> $LnkMt5"
    Write-Host "     $mt5"
} else {
    Write-Host "[WARN] MT5 tidak ditemukan otomatis."
    Write-Host "       Jalankan ulang dengan path manual, contoh:"
    Write-Host "       .\scripts\setup_autostart.ps1 -Mt5Path `"C:\Program Files\MetaTrader 5\terminal64.exe`""
}

Write-Host ""
Write-Host "Folder Startup: $Startup"
Write-Host ""
Write-Host "Langkah berikutnya (agar PC nyala = login + bot):"
Write-Host "  1. Login otomatis Windows: Win+R -> netplwiz -> hapus centang password"
Write-Host "  2. Task Scheduler: XAUUSD_H1_Stage9_Bot (At log on) - sudah terdaftar"
Write-Host "  3. Restart PC, buka MT5 sekali dan centang Simpan password akun trading"
Write-Host "  4. Tes /analisis di Telegram bot Anda"
