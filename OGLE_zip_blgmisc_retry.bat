@echo off
setlocal
set "BASE=K:\altREU-DISCORD\AltREU-Project\Data_downloader_scripts\OGLE"
set "OUT=K:\altREU-DISCORD\AltREU-Project\Data_downloader_scripts\OGLE_zips"

set "ARC="
set "MODE="
if exist "%ProgramFiles%\7-Zip\7z.exe"        ( set "ARC=%ProgramFiles%\7-Zip\7z.exe"        & set "MODE=7z" )
if not defined ARC if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" ( set "ARC=%ProgramFiles(x86)%\7-Zip\7z.exe" & set "MODE=7z" )
if not defined ARC if exist "%ProgramFiles%\WinRAR\WinRAR.exe" ( set "ARC=%ProgramFiles%\WinRAR\WinRAR.exe" & set "MODE=rar" )
if not defined ARC if exist "%ProgramFiles(x86)%\WinRAR\WinRAR.exe" ( set "ARC=%ProgramFiles(x86)%\WinRAR\WinRAR.exe" & set "MODE=rar" )
if not defined ARC ( echo Could not find 7-Zip or WinRAR. & pause & exit /b 1 )

echo Using %MODE%: "%ARC%"
cd /d "%BASE%" || ( echo Could not cd to %BASE% & pause & exit /b 1 )

if exist "%OUT%\blg_misc.zip" del /q "%OUT%\blg_misc.zip"

echo.
echo --- Building blg_misc.zip (blg minus ecl) - full output shown below ---
echo.

if "%MODE%"=="7z" (
  "%ARC%" a -tzip -mx=6 -bsp1 "%OUT%\blg_misc.zip" "OCVS\OCVS_full\blg" -xr!ecl
) else (
  "%ARC%" a -afzip -r -m3 "%OUT%\blg_misc.zip" "OCVS\OCVS_full\blg" -x"OCVS\OCVS_full\blg\ecl\*" -x"OCVS\OCVS_full\blg\ecl"
)

echo.
echo --- Exit code: %errorlevel% ---
echo.

if exist "%OUT%\blg_misc.zip" (
  echo Archive created. Testing integrity...
  "%ARC%" t "%OUT%\blg_misc.zip"
  echo.
  echo --- Test exit code: %errorlevel% ---
) else (
  echo.
  echo NO FILE WAS CREATED. Scroll up to see the error message from %MODE% above.
  echo Common causes: path-length limit (Windows 260-char default^), disk space, or a
  echo permission issue on one of the files inside blg. Copy the error text and send
  echo it back so we can fix the exact cause.
)

echo.
pause
