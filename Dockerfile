FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# invoke via sh explicitly rather than ./entrypoint.sh -- avoids Windows
# bind-mount execute-permission issues when this is later mounted over
# by docker-compose's volume
ENTRYPOINT ["/bin/sh", "entrypoint.sh"]
