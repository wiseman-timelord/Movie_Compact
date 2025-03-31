@echo off

:: Initialization
title Batch Launcher
cd /d "%~dp0"
color 80

:: Globals
set PROG_NAME=Movie Compact

:: Check for Python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not added to the PATH.
    pause
    exit /b 1
)

goto :menu

:printtitle
set DISP_PHASE=%~1
echo =================================================================
echo.    %PROG_NAME%: %DISP_PHASE%
echo =================================================================
goto :eof

:menu
cls
call :printtitle "Main Menu"
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo    1. Run Main Program
echo.
echo    2. Run Installer Tool
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo ----------------------------------------------------------------
set /p choice=Select; Launch Options = 1-2, Exit Launcher = X: 

if /i "%choice%"=="1" goto run_program
if /i "%choice%"=="2" goto run_installer
if /i "%choice%"=="X" goto exit

echo Invalid choice, please try again.
timeout /t 3 /nobreak >nul
goto menu

:run_program
cls
call :printtitle "Launching..."
echo.
timeout /t 2 /nobreak >nul
title Batch Launcher
@echo on
call :activate_venv
python launcher.py
call :deactivate_venv
@echo off
timeout /t 5 /nobreak >nul
color 80
goto menu

:run_installer
cls
call :printtitle "Installing..."
echo.
timeout /t 2 /nobreak >nul
@echo on
call :activate_venv
python requisites.py
call :deactivate_venv
@echo off
timeout /t 5 /nobreak >nul
color 80
goto menu

:activate_venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
exit /b

:deactivate_venv
if exist venv\Scripts\deactivate.bat (
    call venv\Scripts\deactivate.bat
)
exit /b

:exit
exit