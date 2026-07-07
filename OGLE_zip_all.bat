@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  OGLE archiver - zips each subfolder for safe GDrive upload
REM  Produces one .zip per folder into ...\OGLE_zips
REM  Splitting the giant blg\ecl folder (~790k files) by OGLE phase
REM  Every archive is integrity-tested at the end.
REM ============================================================

set "BASE=K:\altREU-DISCORD\AltREU-Project\Data_downloader_scripts\OGLE"
set "OUT=K:\altREU-DISCORD\AltREU-Project\Data_downloader_scripts\OGLE_zips"

REM ---- locate an archiver (7-Zip preferred, then WinRAR) ----
set "ARC="
set "MODE="
if exist "%ProgramFiles%\7-Zip\7z.exe"        ( set "ARC=%ProgramFiles%\7-Zip\7z.exe"        & set "MODE=7z" )
if not defined ARC if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" ( set "ARC=%ProgramFiles(x86)%\7-Zip\7z.exe" & set "MODE=7z" )
if not defined ARC if exist "%ProgramFiles%\WinRAR\WinRAR.exe" ( set "ARC=%ProgramFiles%\WinRAR\WinRAR.exe" & set "MODE=rar" )
if not defined ARC if exist "%ProgramFiles(x86)%\WinRAR\WinRAR.exe" ( set "ARC=%ProgramFiles(x86)%\WinRAR\WinRAR.exe" & set "MODE=rar" )
if not defined ARC (
  echo.
  echo   Could not find 7-Zip or WinRAR in the usual locations.
  echo   Install 7-Zip ^(https://www.7-zip.org^) or WinRAR, then run this again.
  echo.
  pause & exit /b 1
)
echo Using %MODE%:  "%ARC%"
echo Source: %BASE%
echo Output: %OUT%
echo.

if not exist "%BASE%" ( echo ERROR: source folder not found. & pause & exit /b 1 )

REM ---- fresh output folder (clears any leftover/partial files) ----
if not exist "%OUT%" ( mkdir "%OUT%" ) else ( del /q "%OUT%\*" 2>nul )

cd /d "%BASE%"

echo ==================== BUILDING ARCHIVES ====================

REM --- EWS, one per year ---
call :ZIP EWS_2022.zip "EWS\2022-2026\2022"
call :ZIP EWS_2023.zip "EWS\2022-2026\2023"
call :ZIP EWS_2024.zip "EWS\2022-2026\2024"
call :ZIP EWS_2025.zip "EWS\2022-2026\2025"
call :ZIP EWS_2026.zip "EWS\2022-2026\2026"

REM --- standard OCVS categories ---
call :ZIP OCVS_BLAP.zip "OCVS\OCVS_full\BLAP"
call :ZIP OCVS_CBO.zip  "OCVS\OCVS_full\CBO"
call :ZIP OCVS_CV.zip   "OCVS\OCVS_full\CV"
call :ZIP OCVS_Cepheid_Misclassifications.zip "OCVS\OCVS_full\Cepheid_Misclassifications"
call :ZIP OCVS_M54.zip  "OCVS\OCVS_full\M54"
call :ZIP OCVS_gal.zip  "OCVS\OCVS_full\gal"
call :ZIP OCVS_gd.zip   "OCVS\OCVS_full\gd"
call :ZIP OCVS_smc.zip  "OCVS\OCVS_full\smc"
call :ZIP OCVS_lmc.zip  "OCVS\OCVS_full\lmc"

REM --- blg: everything except the huge ecl subtree ---
call :ZIPMISC

REM --- blg\ecl: loose metadata files, then split by OGLE phase ---
call :ZIPMETA
call :ZIP blg_ecl_ogle2.zip "OCVS\OCVS_full\blg\ecl\phot_ogle2"
call :ZIP blg_ecl_ogle3.zip "OCVS\OCVS_full\blg\ecl\phot_ogle3"
call :ZIP blg_ecl_ogle4.zip "OCVS\OCVS_full\blg\ecl\phot_ogle4"

echo.
echo ==================== VERIFYING ARCHIVES ====================
set "FAILED="
for %%F in ("%OUT%\*.zip") do (
  echo Testing %%~nxF ...
  "%ARC%" t "%%F" >nul 2>&1
  if errorlevel 1 ( echo    FAILED: %%~nxF & set "FAILED=1" ) else ( echo    OK )
)

echo.
echo ==================== DONE ====================
dir /b "%OUT%\*.zip"
echo.
if defined FAILED (
  echo One or more archives FAILED verification - see above. Re-run to rebuild.
) else (
  echo All archives built and verified OK. Upload the OGLE_zips folder to Google Drive.
)
echo.
pause
exit /b 0

:ZIP
REM  %1 = archive name   %2 = source folder (relative to BASE)
echo.
echo --- %~1 ---
if "%MODE%"=="7z" (
  "%ARC%" a -tzip -mx=6 -bso0 -bsp2 "%OUT%\%~1" "%~2"
) else (
  "%ARC%" a -afzip -r -m3 -ep1 "%OUT%\%~1" "%~2"
)
exit /b

:ZIPMISC
echo.
echo --- blg_misc.zip (blg minus ecl) ---
if "%MODE%"=="7z" (
  "%ARC%" a -tzip -mx=6 -bso0 -bsp2 "%OUT%\blg_misc.zip" "OCVS\OCVS_full\blg" -xr!ecl
) else (
  "%ARC%" a -afzip -r -m3 "%OUT%\blg_misc.zip" "OCVS\OCVS_full\blg" -x"OCVS\OCVS_full\blg\ecl\*" -x"OCVS\OCVS_full\blg\ecl"
)
exit /b

:ZIPMETA
echo.
echo --- blg_ecl_meta.zip (loose files in blg\ecl) ---
pushd "%BASE%\OCVS\OCVS_full\blg\ecl"
if "%MODE%"=="7z" (
  "%ARC%" a -tzip -mx=6 -bso0 "%OUT%\blg_ecl_meta.zip" README ecl.dat ell.dat ident.dat paper.pdf remarks.txt
) else (
  "%ARC%" a -afzip -m3 "%OUT%\blg_ecl_meta.zip" README ecl.dat ell.dat ident.dat paper.pdf remarks.txt
)
popd
exit /b
