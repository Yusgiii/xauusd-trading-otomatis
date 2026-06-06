# Daftarkan tugas Windows Task Scheduler untuk bot XAUUSD Stage 9.
# Jalankan PowerShell sebagai Administrator:
#   Set-ExecutionPolicy -Scope Process Bypass
#   cd "D:\trading\trading otomatis 5"
#   .\scripts\register_windows_tasks.ps1
#
# Retrain harian (opsional — auto-retrain di service biasanya cukup):
#   .\scripts\register_windows_tasks.ps1 -IncludeDailyRetrain

param(
    [switch]$IncludeDailyRetrain
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
$stage9Script = Join-Path $projectDir "scripts\stage9_service.py"
$startupBat = Join-Path $projectDir "scripts\startup_service.bat"
$pipelineScript = Join-Path $projectDir "run_pipeline.py"

if (-not (Test-Path $python)) {
    throw "Python venv tidak ditemukan: $python"
}
if (-not (Test-Path $stage9Script)) {
    throw "Script tidak ditemukan: $stage9Script"
}
if (-not (Test-Path $startupBat)) {
    throw "Startup script tidak ditemukan: $startupBat"
}

$TaskBot = "XAUUSD_Stage9_Bot"
$TaskRetrain = "XAUUSD_Daily_Retrain"
$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"

function Remove-LegacyTasks {
    $legacyNames = @(
        "XAUUSD_H1_Stage9_Bot",
        "XAUUSD_H1_Task",
        "GBPJPY_Stage9_Bot",
        "GBPJPY_Weekly_Retrain"
    )
    foreach ($name in $legacyNames) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    }
    Get-ScheduledTask -ErrorAction SilentlyContinue |
        Where-Object { $_.TaskName -like "XAUUSD_H1*" } |
        ForEach-Object {
            Write-Host "[CLEANUP] Hapus task lama: $($_.TaskName)"
            Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false -ErrorAction SilentlyContinue
        }
    $remaining = @(Get-ScheduledTask -ErrorAction SilentlyContinue |
        Where-Object { $_.TaskName -like "XAUUSD_H1*" })
    if ($remaining.Count -gt 0) {
        Write-Host "[WARN] Task lama masih ada (butuh Admin untuk hapus): $($remaining.TaskName -join ', ')"
        Write-Host "       Jalankan: scripts\install_tasks_admin.bat (klik kanan -> Run as administrator)"
    }
}

function New-ServiceTaskSettings {
    New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero)
}

Remove-LegacyTasks

# --- Task 1: Bot Stage 9 saat login ---
Unregister-ScheduledTask -TaskName $TaskBot -Confirm:$false -ErrorAction SilentlyContinue

# Pakai startup_service.bat: delay MT5, hapus lock stale, log ke logs/startup.log
$actionBot = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"`"$startupBat`"`"" `
    -WorkingDirectory $projectDir

$triggerBot = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$triggerBot.Delay = "PT60S"
$settingsBot = New-ServiceTaskSettings

function Register-TradingTask {
    param(
        [string]$Name,
        [Microsoft.Management.Infrastructure.CimInstance]$Action,
        [Microsoft.Management.Infrastructure.CimInstance[]]$Trigger,
        [Microsoft.Management.Infrastructure.CimInstance]$Settings,
        [string]$Description
    )
    try {
        Register-ScheduledTask `
            -TaskName $Name `
            -Action $Action `
            -Trigger $Trigger `
            -Settings $Settings `
            -RunLevel Highest `
            -User $CurrentUser `
            -Description $Description `
            -Force | Out-Null
        Write-Host "[OK] Tugas terdaftar (RunLevel Highest): $Name"
    } catch {
        Write-Host "[WARN] RunLevel Highest gagal ($($_.Exception.Message)); coba Limited..."
        Register-ScheduledTask `
            -TaskName $Name `
            -Action $Action `
            -Trigger $Trigger `
            -Settings $Settings `
            -RunLevel Limited `
            -User $CurrentUser `
            -Description $Description `
            -Force | Out-Null
        Write-Host "[OK] Tugas terdaftar (RunLevel Limited): $Name"
    }
}

Register-TradingTask `
    -Name $TaskBot `
    -Action $actionBot `
    -Trigger @($triggerBot) `
    -Settings $settingsBot `
    -Description "Bot Telegram XAUUSD Stage 9 (moment alert + /analisa). MT5 harus login. Pakai --latest-run."

Write-Host "     Trigger: At log on + delay 60s (PT60S)"
Write-Host "     Execute: cmd.exe /c $startupBat"
Write-Host "     WorkDir: $projectDir"

# --- Task 2: Retrain harian (opsional) ---
Unregister-ScheduledTask -TaskName $TaskRetrain -Confirm:$false -ErrorAction SilentlyContinue

if ($IncludeDailyRetrain) {
    if (-not (Test-Path $pipelineScript)) {
        throw "Script tidak ditemukan: $pipelineScript"
    }
    $actionRetrain = New-ScheduledTaskAction `
        -Execute $python `
        -Argument "`"$pipelineScript`"" `
        -WorkingDirectory $projectDir

    # 05:00 WIB = waktu lokal PC (asumsikan timezone WIB)
    $triggerRetrain = New-ScheduledTaskTrigger -Daily -At "05:00"
    $settingsRetrain = New-ServiceTaskSettings

    Register-TradingTask `
        -Name $TaskRetrain `
        -Action $actionRetrain `
        -Trigger @($triggerRetrain) `
        -Settings $settingsRetrain `
        -Description "Retrain harian run_pipeline.py (05:00 WIB). Nonaktifkan jika auto-retrain service cukup."

    Write-Host "     Trigger: Daily 05:00 lokal (~22:00 UTC)"
} else {
    Write-Host "[SKIP] $TaskRetrain tidak didaftarkan (auto-retrain di service aktif)."
    Write-Host "       Untuk aktifkan: .\scripts\register_windows_tasks.ps1 -IncludeDailyRetrain"
}

try {
    Start-ScheduledTask -TaskName $TaskBot -ErrorAction Stop
    Write-Host "[OK] Tugas dijalankan sekarang: $TaskBot"
} catch {
    Write-Host "[WARN] Tugas terdaftar tetapi gagal start otomatis: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Cek: taskschd.msc -> Task Scheduler Library"
Write-Host "Hapus semua: .\scripts\unregister_windows_tasks.ps1"
