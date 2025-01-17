# Wybieramy obraz bazowy Pythona
FROM python:3.10-slim

# Ustawienie katalogu roboczego
WORKDIR ./

# Kopiowanie requirements.txt i instalowanie zależności
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie wszystkich plików aplikacji
COPY . .

# Rozpakowywanie plików .gz
RUN gunzip -k data/*.gz

# Uruchomienie aplikacji Flask
EXPOSE 8000

CMD ["python", "-u", "app.py"]
