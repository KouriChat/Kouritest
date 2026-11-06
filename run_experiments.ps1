[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Tasks = "global,csc,dzh,seto,arctic01,hecate",
    [switch]$Fulfill,
    [int]$ExtraGroup = 0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "run_map_experiments.py"
$DefaultTasks = "global,csc,dzh,seto,arctic01,hecate"
$AllowedTasks = @("global", "csc", "dzh", "seto", "arctic01", "hecate")

function Resolve-TaskSelection([string]$Value) {
    $trimmed = $Value.Trim()
    if ($trimmed -eq "") { return $DefaultTasks }
    if ($trimmed.ToLowerInvariant() -eq "all") { return $DefaultTasks }
    $items = @($trimmed.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    if ($items.Count -eq 0) { return $DefaultTasks }
    foreach ($item in $items) {
        if ($item.ToLowerInvariant() -eq "arctic02") {
            throw "arctic02 is retired. Use arctic01 (the 0.1-degree Arctic task) instead."
        }
        if ($AllowedTasks -notcontains $item) {
            throw "Unknown task '$item'. Allowed: $($AllowedTasks -join ', ')."
        }
    }
    return ($items -join ",")
}

if (-not $CliArgs) { $CliArgs = @() }
$taskArgIndex = -1
for ($i = 0; $i -lt $CliArgs.Count; $i++) {
    if ($CliArgs[$i] -eq "--tasks") { $taskArgIndex = $i; break }
    if ($CliArgs[$i] -like "--tasks=*") { $taskArgIndex = $i; break }
}

if ($taskArgIndex -ge 0) {
    if ($CliArgs[$taskArgIndex] -eq "--tasks") {
        if ($taskArgIndex + 1 -ge $CliArgs.Count) {
            throw "--tasks requires a value."
        }
        $CliArgs[$taskArgIndex + 1] = Resolve-TaskSelection $CliArgs[$taskArgIndex + 1]
    } else {
        $value = $CliArgs[$taskArgIndex].Substring("--tasks=".Length)
        $CliArgs[$taskArgIndex] = "--tasks=" + (Resolve-TaskSelection $value)
    }
} else {
    $selected = Resolve-TaskSelection $Tasks
    $CliArgs = @("--tasks", $selected) + $CliArgs
}

if ($ExtraGroup -lt 0) {
    throw "-ExtraGroup must be >= 0."
}
if ($Fulfill -and $ExtraGroup -gt 0) {
    throw "-Fulfill and -ExtraGroup cannot be used together."
}

# Normalize the Python-style --fulfill / --extra-group options into the
# PowerShell parameters.  A generic list avoids PowerShell range edge cases
# when the removed option is the first or last argument.
$extraGroupValue = $ExtraGroup
$normalizedArgs = [System.Collections.Generic.List[string]]::new()
foreach ($arg in $CliArgs) { $normalizedArgs.Add($arg) }
for ($i = 0; $i -lt $normalizedArgs.Count; $i++) {
    if ($normalizedArgs[$i] -eq "--fulfill") {
        $Fulfill = $true
        $normalizedArgs.RemoveAt($i)
        break
    }
    if ($normalizedArgs[$i] -like "--extra-group=*") {
        $extraGroupValue = [int]$normalizedArgs[$i].Substring("--extra-group=".Length)
        $normalizedArgs.RemoveAt($i)
        break
    }
    if ($normalizedArgs[$i] -eq "--extra-group") {
        if ($i + 1 -ge $normalizedArgs.Count) { throw "--extra-group requires a value." }
        $extraGroupValue = [int]$normalizedArgs[$i + 1]
        $normalizedArgs.RemoveAt($i + 1)
        $normalizedArgs.RemoveAt($i)
        break
    }
}
$CliArgs = @($normalizedArgs)

if ($extraGroupValue -lt 0) {
    throw "--extra-group must be >= 0."
}
if ($Fulfill -and $extraGroupValue -gt 0) {
    throw "--fulfill and --extra-group cannot be used together."
}
if ($Fulfill) {
    $CliArgs = @("--fulfill") + $CliArgs
}
if ($extraGroupValue -gt 0) {
    $CliArgs = @("--extra-group", "$extraGroupValue") + $CliArgs
}

$python = $env:MAP_EXPERIMENT_PYTHON
if (-not $python) {
    $bundled = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $bundled) {
        $python = $bundled
    }
}
if (-not $python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source }
}
if (-not $python) {
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source }
}
if (-not $python) {
    throw "Python not found. Set MAP_EXPERIMENT_PYTHON to a Python with numpy and Pillow."
}

& $python $script @CliArgs
exit $LASTEXITCODE
