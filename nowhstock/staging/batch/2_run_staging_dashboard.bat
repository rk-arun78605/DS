@echo off
REM ============================================================
REM RUN STAGING DASHBOARD
REM Location: D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch\
REM Purpose: Launch Streamlit app with staging views
REM ============================================================

echo ================================================================================
echo LAUNCHING STAGING DASHBOARD
echo ================================================================================
echo.
echo Environment: STAGING (Testing Priority-to-Priority Transfers)
echo Views: mv_recommendations_complete_staging
echo Port: 8502 (different from production 8501)
echo.
echo This dashboard will:
echo   - Use staging materialized views
echo   - Show priority shops as sources
echo   - Allow priority-to-priority transfers
echo   - Keep production completely untouched
echo.

cd /d "%~dp0.."

echo Starting Streamlit on port 8502...
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run nowhstock_ds_STAGING.py --server.port 8502

pause
