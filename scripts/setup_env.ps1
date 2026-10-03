# Create .venv in the repository and install the package with its dev tools.
# Usage (repository root): powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# Interpreter: $env:PYTHON if set, else python on PATH (Python 3.12 or newer).
$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $py -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -e ".[dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -c "import dynamic_trading_engine, numpy, scipy, cvxpy; print('ready:', dynamic_trading_engine.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'cvxpy', cvxpy.__version__)"
exit $LASTEXITCODE
