@echo off
echo Starting Urdu Transformer Chatbot...
echo ====================================
echo.
echo Installing dependencies (if needed)...
pip install -r requirements.txt
echo.
echo Starting Streamlit app...
streamlit run app.py
pause
