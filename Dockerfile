FROM python:3.10-slim

# Instalacja zależności
WORKDIR ./
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie aplikacji
COPY . .

RUN gunzip -k data/*.gz


# Uruchamianie aplikacji
CMD ["python", "-u", "app.py"]
