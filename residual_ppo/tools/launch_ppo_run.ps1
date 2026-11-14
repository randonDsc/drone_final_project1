# Directory: <repo root>
# Launch a PPO training run (training/train_path_ppo.py) in its OWN PowerShell window, detached from VS Code /
# Claude Code, with the console written to runs/ppo_path/<id>/logs/console_<stamp>.log. Returns right away.
#
#   $env:ISAAC_PYTHON = "<Isaac Sim>\python.bat"          # or pass -IsaacPython
#   & residual_ppo\tools\launch_ppo_run.ps1 -RunId ppo_s0_1p5m -Mode fresh
#   & residual_ppo\tools\launch_ppo_run.ps1 -RunId ppo_s0_1p5m -Mode resume `
#       -ResumeModel residual_ppo/runs/ppo_path/ppo_s0_1p5m/checkpoints/ppo_250000_steps.zip
#
# Stop a run ONLY by creating runs/ppo_path/<id>/STOP. Ctrl+C or closing the window kills it (nothing saved);
# recovery is then a resume from the last checkpoint. Success = the console log ends with "RESULT: DONE" and
# status.json says "done"; never judge by an exit code.
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][ValidateSet("fresh", "resume")][string]$Mode,
    [string]$ResumeModel = "",
    [string[]]$ExtraArgs = @(),
    [string]$IsaacPython = $env:ISAAC_PYTHON
)
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $IsaacPython) { throw "Isaac python.bat not given: pass -IsaacPython or set `$env:ISAAC_PYTHON" }
if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) { throw "Isaac python.bat not found: $IsaacPython" }
if ($Mode -eq "resume" -and -not $ResumeModel) { throw "-Mode resume needs -ResumeModel" }
if ($Mode -eq "fresh" -and $ResumeModel) { throw "-ResumeModel is only for -Mode resume" }
if ($RunId -notmatch '^[A-Za-z0-9_]+$') { throw "RunId may contain only letters, digits and _" }

$runDir = Join-Path $repo "residual_ppo\runs\ppo_path\$RunId"
$logDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "console_${Mode}_$stamp.log"

function Quote([string]$s) { "'" + $s.Replace("'", "''") + "'" }
$pyArgs = @("residual_ppo\training\train_path_ppo.py", "--run-id", $RunId, "--mode", $Mode)
if ($ResumeModel) { $pyArgs += @("--resume-model", $ResumeModel) }
$pyArgs += $ExtraArgs
$argText = ($pyArgs | ForEach-Object { Quote $_ }) -join " "
$cmd = "Set-Location -LiteralPath $(Quote $repo); & $(Quote $IsaacPython) $argText *> $(Quote $log)"

$p = Start-Process powershell.exe -ArgumentList "-NoProfile", "-Command", $cmd -PassThru
$rel = $log.Substring($repo.Length + 1)
"launched: run $RunId, mode $Mode (window process id $($p.Id))"
"console log: $rel"
"watch:  Get-Content $rel -Wait -Tail 5"
"status: Get-Content residual_ppo\runs\ppo_path\$RunId\status.json"
"stop:   New-Item -ItemType File residual_ppo\runs\ppo_path\$RunId\STOP"
