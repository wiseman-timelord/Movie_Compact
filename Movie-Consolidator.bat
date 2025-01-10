@echo off
setlocal enabledelayedexpansion
title MovieConsolidate-Batch
color 80
echo Initialization Complete.
timeout /t 1 >nul

set "PYTHON_VERSION_NODECIMAL=312"
set "PYTHON_EXE_TO_USE="

goto :main_logic

:printHeader
echo ========================================================================================================================
echo    %~1
echo ========================================================================================================================
goto :eof

:printSeparator
echo ========================================================================================================================
goto :eof

:main_logic
set "ScriptDirectory=%~dp0"
set "ScriptDirectory=%ScriptDirectory:~0,-1%"
cd /d "%ScriptDirectory%"
echo Dp0'd to Script.

net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Error: Admin Required!
    timeout /t 2 >nul
    echo Right Click, Run As Administrator.
    timeout /t 2 >nul
    goto :end_of_script
)
echo Status: Administrator
timeout /t 1 >nul

for %%I in (
    "C:\Python%PYTHON_VERSION_NODECIMAL%\python.exe"
    "C:\Program Files\Python%PYTHON_VERSION_NODECIMAL%\python.exe"
    "%LocalAppData%\Programs\Python\Python%PYTHON_VERSION_NODECIMAL%\python.exe"
) do (
    if exist "%%~I" (
        set "PYTHON_EXE_TO_USE=%%~dpI\python.exe"
        goto :found_python
    )
)
echo Error: Python %PYTHON_VERSION_NODECIMAL% not found. Please install Python %PYTHON_VERSION_NODECIMAL%.
goto :end_of_file
:found_python
echo Python %PYTHON_VERSION_NODECIMAL% found.
echo Using `python.exe` from: %PYTHON_EXE_TO_USE%
echo.

:check_venv
if not exist "venv" (
    echo Creating virtual environment...
    "%PYTHON_EXE_TO_USE%" -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        goto :end_of_file
    )
)
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    goto :end_of_file
)

:main_menu
cls
call :printHeader "MovieConsolidate-Batch"
echo.
echo.
echo     1. Run MovieConsolidate
echo.
echo     2. Install Requirements
echo.
echo.
echo.
call :printSeparator
set /p choice=Selection; Menu Options = 1-2, Exit MovieConsolidate-Batch = X: 

if "!choice!"=="1" (
    goto run_launcher
    echo Selected: Run MovieConsolidate
    timeout /t 1 >nul
) else if "!choice!"=="2" (
    echo Selected: Install Requirements
    timeout /t 1 >nul
    goto install_requirements
) else if /i "!choice!"=="X" (
    echo Selected: Exit MovieConsolidate-Batch
    timeout /t 1 >nul
    goto :end_of_file
) else (
    echo Invalid option. Please try again.
    pause
    goto :main_menu
)

:run_launcher
cls
call :printHeader "Run MovieConsolidate"
echo.
echo Running MovieConsolidate...
echo Using Python executable: %PYTHON_EXE_TO_USE%
echo.

echo Checking Python version...
python --version
echo.

echo Listing installed packages...
pip list
echo.

echo Launching launcher.py...
python .\launcher.py
pause
goto exit

:install_requirements
cls
call :printHeader "Install Requirements"
echo.
echo Initiating installation via installer.py...
echo.

python .\installer.py
if errorlevel 1 (
    echo Failed to install requirements.
) else (
    echo Requirements installed successfully.
)
echo.

echo Installation complete
echo.
echo Please review the output above.
echo.
pause
goto :main_menu

:end_of_file
cls
call :printHeader "Exit MovieConsolidate-Batch"
echo.
timeout /t 1 >nul
echo Exiting MovieConsolidate-Batch
timeout /t 1 >nul
echo All processes finished.
timeout /t 1 >nul
if "%VIRTUAL_ENV%" NEQ "" (
    echo Deactivating virtual environment...
    call venv\Scripts\deactivate
)
exit /b