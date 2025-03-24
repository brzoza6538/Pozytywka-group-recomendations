FROM python:3.10-slim

WORKDIR .

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . ./

RUN find /data -name "*.gz" -exec gunzip -k {} \; || true

EXPOSE 8000

# Uruchomienie aplikacji
CMD ["python", "-u", "main.py"]
# CMD ["ls", "-al", "./"]
