@echo OFF
REM Batch script to run nightly WNBA Betting Recommender tasks

REM Get the directory of this batch script (backend/)
SET "SCRIPT_DIR=%~dp0"

REM Navigate to the project root directory (one level up from backend/)
CD /D "%SCRIPT_DIR%..\"

ECHO Activating virtual environment...
CALL .\backend\venv\Scripts\activate.bat

IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment. Exiting.
    EXIT /B 1
)

ECHO Running WNBA stats scraper...
python .\backend\utils\wnba_stats_scraper.py
IF ERRORLEVEL 1 (
    ECHO WNBA stats scraper failed.
    REM Decide if you want to exit or continue with other scripts
)

ECHO Running odds scraper...
python .\backend\utils\odds_scraper.py
IF ERRORLEVEL 1 (
    ECHO Odds scraper failed.
    REM Decide if you want to exit or continue with other scripts
)

ECHO Running model training script...
python .\backend\models\train_model.py
IF ERRORLEVEL 1 (
    ECHO Model training script failed.
)

ECHO Deactivating virtual environment (if applicable)...
REM Deactivation is often handled by the CALL to activate.bat or simply by script end
REM If you have a specific deactivate command for your venv, you can add it here.

ECHO Nightly tasks finished.
EXIT /B 0 