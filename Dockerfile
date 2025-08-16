FROM python:3.11-slim


WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "deepfake:app", "--host", "0.0.0.0", "--port", "5000"]
