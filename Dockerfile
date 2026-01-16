FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + glossary
COPY main.py .
COPY glossary_rollomatic.csv .

ENV GLOSSARY_PATH=glossary_rollomatic.csv

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
