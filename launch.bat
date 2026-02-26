@echo off
REM Launch BG-cleaner with GPU support on Windows.
REM The Python code registers NVIDIA DLL directories automatically,
REM but this script also sets them in PATH as a fallback.

call .venv\Scripts\activate.bat

REM Add NVIDIA pip-installed DLLs to PATH (if present).
for %%D in (cudnn cublas cuda_runtime cufft) do (
    if exist ".venv\Lib\site-packages\nvidia\%%D\bin" (
        set "PATH=.venv\Lib\site-packages\nvidia\%%D\bin;%PATH%"
    )
)

streamlit run src/bgcleaner/ui/app.py