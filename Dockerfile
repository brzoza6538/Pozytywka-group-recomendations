FROM python:3.10-slim

# Instalacja zależności
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie aplikacji
COPY . .

RUN gunzip data/*.gz


# Uruchamianie aplikacji
CMD ["python", "app.py"]
