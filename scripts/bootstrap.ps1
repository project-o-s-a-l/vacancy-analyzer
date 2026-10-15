param(
    [string]$EnvironmentPath = '.local-tmp/dev-env',
    [ValidateSet('base', 'ml', 'semantic', 'all')]
    [string]$Extras = 'all',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & $Python -c "import sys, platform; assert sys.version_info[:2] == (3, 13) and sys.platform == 'win32' and platform.machine().upper() == 'AMD64', 'Use CPython 3.13 x64 on Windows for this lock'"
    if ($LASTEXITCODE -ne 0) { throw 'Unsupported interpreter for requirements/windows-py313.txt' }
    if ((Test-Path -LiteralPath $EnvironmentPath) -and
        -not (Test-Path -LiteralPath (Join-Path $EnvironmentPath 'pyvenv.cfg'))) {
        throw 'EnvironmentPath exists but is not a virtual environment; choose a new path'
    }
    if (-not (Test-Path -LiteralPath $EnvironmentPath)) {
        & $Python -m venv $EnvironmentPath
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    }
    $environmentPython = Join-Path $EnvironmentPath 'Scripts/python.exe'
    & $environmentPython -c "import sys; assert sys.version_info[:2] == (3, 13), 'Existing environment must use Python 3.13'"
    if ($LASTEXITCODE -ne 0) { throw 'Use a new EnvironmentPath; existing environment is unusable' }
    $constraints = Join-Path $projectRoot 'requirements/windows-py313.txt'
    & $environmentPython -m pip install -c $constraints 'pip==25.2' 'setuptools==84.2.1'
    if ($LASTEXITCODE -ne 0) { throw 'Toolchain installation failed' }
    $package = switch ($Extras) {
        'base' { '.' }
        'all' { '.[ml,semantic]' }
        default { ".[$Extras]" }
    }
    & $environmentPython -m pip install --no-build-isolation -c $constraints -e $package --group dev
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
    & $environmentPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed' }
    Write-Output "Ready: $environmentPython (extras=$Extras)"
} finally {
    Pop-Location
}
