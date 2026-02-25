@echo off
echo ========================================
echo EV Research Pipeline - Run All Phases
echo ========================================
echo.

REM Change to project root
cd /d "%~dp0\.."

REM Phase 1: Model Check
echo [PHASE 1] Running Model Verification Tests...
.\venv\Scripts\python pipeline\phase_1_model_check\test_models.py
if %errorlevel% neq 0 (
    echo [ERROR] Phase 1 failed. Fix model issues before continuing.
    exit /b 1
)
echo [OK] Phase 1 complete
echo.

REM Phase 2: Ingestion
echo [PHASE 2] Running Data Ingestion Tests...
.\venv\Scripts\python pipeline\phase_2_ingestion\test_ingestion.py
if %errorlevel% neq 0 (
    echo [ERROR] Phase 2 failed. Fix data issues before continuing.
    exit /b 1
)
echo [OK] Phase 2 complete
echo.

REM Phase 3: Chunking
echo [PHASE 3] Running Chunking Tests...
.\venv\Scripts\python pipeline\phase_3_chunking\test_chunking.py
if %errorlevel% neq 0 (
    echo [ERROR] Phase 3 failed. Fix chunking issues before continuing.
    exit /b 1
)
echo [OK] Phase 3 complete
echo.

REM Phase 4: Retrieval
echo [PHASE 4] Running Retrieval Tests...
.\venv\Scripts\python pipeline\phase_4_retrieval\test_retrieval.py
if %errorlevel% neq 0 (
    echo [ERROR] Phase 4 failed. Fix retrieval issues before continuing.
    exit /b 1
)
echo [OK] Phase 4 complete
echo.

REM Phase 5: Full Pipeline
echo [PHASE 5] Running Full Pipeline with Model Responses...
.\venv\Scripts\python pipeline\phase_5_full_pipeline\run_pipeline.py
if %errorlevel% neq 0 (
    echo [ERROR] Phase 5 failed.
    exit /b 1
)
echo [OK] Phase 5 complete
echo.

echo ========================================
echo ALL PHASES COMPLETED SUCCESSFULLY!
echo ========================================
echo Results saved to: test_results.xlsx
echo.
pause
