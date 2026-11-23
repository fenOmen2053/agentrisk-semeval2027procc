@echo off
setlocal
where python >nul 2>nul
if errorlevel 1 goto use_py
python -X utf8 -B "%~dp0main.py" %*
goto done
:use_py
py -3 -X utf8 -B "%~dp0main.py" %*
:done
set "AGENTRISK_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %AGENTRISK_EXIT%
