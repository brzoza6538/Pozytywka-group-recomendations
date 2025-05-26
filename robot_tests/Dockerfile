FROM python:3.11-slim

USER root
WORKDIR /app

# Update i instalacja pakietów potrzebnych do robot framework i selenium
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    default-jre-headless \
    unzip \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Instalacja przeglądarki Chromium i chromedriver nie jest konieczna, bo selenium jest w osobnym serwisie
# (możesz usunąć tę linię, bo netcat jest już zainstalowany powyżej)
# RUN apt-get update && apt-get install -y netcat

# Instalacja Pythona i pip oraz robot framework i selenium library
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8002

CMD bash -c "while ! nc -z selenium 4444; do echo Waiting for Selenium...; sleep 1; done; robot first_page_tests.robot && python3 -m http.server 8002"
