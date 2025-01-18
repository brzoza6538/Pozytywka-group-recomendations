FROM python:3.10-slim

WORKDIR ./

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN gunzip -k data/*.gz

EXPOSE 8000

CMD ["python", "-u", "app.py"]
