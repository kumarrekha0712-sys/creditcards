# Credit Card Fraud Detection

## Run the dashboard at 127.0.0.1

1. Open this project folder.
2. Double-click `start_project.bat`.
3. Open **http://127.0.0.1:8000/** in Chrome.

Do NOT double-click `frontend/index.html`; FastAPI serves it at the 127.0.0.1 URL.

API docs: http://127.0.0.1:8000/docs

If you prefer CMD:
```
cd credit-card-fraud-detection\backend
python -m pip install -r requirements.txt
python train_model.py
python -m uvicorn main:app --reload --port 8000
```

The threshold slider sends POST requests to `/evaluate` and updates accuracy, precision, recall, F1 and the confusion matrix without retraining the model.
