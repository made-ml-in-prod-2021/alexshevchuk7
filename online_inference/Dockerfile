FROM python:3.6
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY partial_gauss.pkl /partial_gauss.pkl
COPY partial_lrg.pkl /partial_lrg.pkl
COPY partial_rfc.pkl /partial_rfc.pkl
COPY model.pkl /model.pkl
COPY app.py /app.py

WORKDIR .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]