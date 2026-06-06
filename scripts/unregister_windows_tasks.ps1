# Hapus tugas terjadwal trading XAUUSD (nama baru + legacy).
$legacyNames = @(
    "XAUUSD_Stage9_Bot",
    "XAUUSD_Daily_Retrain",
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
        Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false
        Write-Host "[HAPUS] $($_.TaskName)"
    }
Write-Host "Task trading XAUUSD dihapus."
