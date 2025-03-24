FROM python:3.10-slim

WORKDIR .

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . ./

EXPOSE 8001

CMD ["python", "-u", "recommendation_service.py"]
# CMD ["ls", "-al", "./"]
