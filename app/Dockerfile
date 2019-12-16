FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

RUN pip install joblib scikit-learn==0.21.2

COPY ./knn.joblib /model/
COPY ./metrics.py /app
COPY ./main.py /app