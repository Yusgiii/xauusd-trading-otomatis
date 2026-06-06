' Jalankan Stage 9 tanpa jendela CMD (Task Scheduler).
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
bat = scriptDir & "\run_stage9_service_task.bat"
CreateObject("WScript.Shell").Run """" & bat & """", 0, False
