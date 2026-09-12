@echo off

cd /d "%~dp0"

call conda init cmd.exe
call conda activate semlab

pip install -r requirements.txt

if errorlevel 1 goto fail

python train_atp3_model.py

if errorlevel 1 goto fail

echo Training complete. Reload Streamlit to use the new model.
goto end

:fail
echo Training failed. Review the error above and README.md.

:end
pause