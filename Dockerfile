FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

ENTRYPOINT ["python", "train.py"]
CMD []
