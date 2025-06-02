@echo OFF
REM Batch script to run nightly WNBA Betting Recommender tasks

REM Get the directory of this batch script (backend/)
SET "SCRIPT_DIR=%~dp0"

REM Navigate to the project root directory (one level up from backend/)
CD /D "%SCRIPT_DIR%..\"

ECHO Setting up date variables using PowerShell...
FOR /F "usebackq" %%i IN (`powershell -NoProfile -Command "(Get-Date).ToString('yyyy')"`) DO SET CURRENT_YEAR=%%i
FOR /F "usebackq" %%i IN (`powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd')"`) DO SET CURRENT_DATE_YYYYMMDD=%%i
FOR /F "usebackq" %%i IN (`powershell -NoProfile -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"`) DO SET YESTERDAY_YYYYMMDD=%%i

ECHO Activating virtual environment...
CALL .\backend\venv\Scripts\activate.bat

IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Exiting.
    EXIT /B 1
)

ECHO Running WNBA stats scraper (to get latest game results for current season)...
python .\backend\utils\wnba_stats_scraper.py --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 (
    ECHO WNBA stats scraper failed.
    REM Consider exiting if this is critical: EXIT /B 1
)

ECHO Running odds scraper (to get latest player props for current day)...
python .\backend\utils\odds_scraper.py --fetch-current
IF ERRORLEVEL 1 (
    ECHO Odds scraper failed.
    REM Consider exiting: EXIT /B 1
)

ECHO Running predictor script (to generate new predictions for current day)...
python .\backend\predict\predictor.py --date %CURRENT_DATE_YYYYMMDD%
IF ERRORLEVEL 1 (
    ECHO Predictor script failed.
    REM Consider exiting: EXIT /B 1
)

ECHO Running prediction outcome processing script (for yesterday's games)...
python .\backend\utils\process_prediction_outcomes.py --date %YESTERDAY_YYYYMMDD%
IF ERRORLEVEL 1 (
    ECHO Prediction outcome processing failed.
)

ECHO Running model training scripts (using potentially updated data for current season)...
ECHO Training for points...
python .\backend\models\train_model.py --target_stat points --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 ( ECHO Points model training failed. )

ECHO Training for rebounds...
python .\backend\models\train_model.py --target_stat rebounds --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 ( ECHO Rebounds model training failed. )

ECHO Training for assists...
python .\backend\models\train_model.py --target_stat assists --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 ( ECHO Assists model training failed. )

ECHO Training for three_pointers_made...
python .\backend\models\train_model.py --target_stat three_pointers_made --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 ( ECHO Three_pointers_made model training failed. )

ECHO Training for steals...
python .\backend\models\train_model.py --target_stat steals --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 ( ECHO Steals model training failed. )

ECHO Training for blocks...
python .\backend\models\train_model.py --target_stat blocks --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 ( ECHO Blocks model training failed. )

ECHO Training for turnovers...
python .\backend\models\train_model.py --target_stat turnovers --seasons %CURRENT_YEAR%
IF ERRORLEVEL 1 ( ECHO Turnovers model training failed. )


ECHO Deactivating virtual environment (if applicable)...
REM Deactivation is often handled by the CALL to activate.bat or simply by script end

ECHO Nightly tasks finished.
EXIT /B 0 