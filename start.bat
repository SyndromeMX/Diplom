@echo off
echo Create venv...
python -m venv venv

echo Venv activate...
call venv\Scripts\activate.bat

echo Install numpy...
pip install matplotlib

echo Installed complete. Check version numpy
pip show numpy

cmd /k