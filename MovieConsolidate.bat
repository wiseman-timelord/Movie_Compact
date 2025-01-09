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
    goto run_movieconsolidate
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

:run_movieconsolidate
cls
call :printHeader "Run MovieConsolidate"
echo.
echo Running MovieConsolidate...
echo Using Python executable: %PYTHON_EXE_TO_USE%
echo.

echo Checking Python version...
"%PYTHON_EXE_TO_USE%" --version
echo.

echo Listing installed packages...
"%PYTHON_EXE_TO_USE%" -m pip list
echo.

echo Launching scripts\interface.py...
"%PYTHON_EXE_TO_USE%" .\scripts\interface.py
pause
goto exit

:install_requirements
cls
call :printHeader "Install Requirements"
echo.
echo Initiating installation via installer.py...
echo.

"%PYTHON_EXE_TO_USE%" .\installer.py
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
exit /b