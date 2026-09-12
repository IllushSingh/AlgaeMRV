@echo off

cd /d "%~dp0"

call conda init cmd.exe
call conda activate semlab

pip install -r requirements.txt

if errorlevel 1 goto fail

if not exist "models\atp3_forecast.joblib" (
    echo Forecast model not found. Training it now...
    python train_atp3_model.py
    if errorlevel 1 goto fail
)

python -m streamlit run app.py

goto end

:fail
echo Setup or launch failed. Review the error above.

:end
pause