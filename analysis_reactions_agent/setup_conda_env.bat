@echo off
setlocal

set ENV_NAME=yieldnet-reaction-agent

call conda env remove -n %ENV_NAME% -y
call conda env create --solver libmamba -f environment.yml
if errorlevel 1 exit /b 1

call conda run -n %ENV_NAME% python verify_chem_stack.py
if errorlevel 1 exit /b 1

echo Environment %ENV_NAME% is ready.
