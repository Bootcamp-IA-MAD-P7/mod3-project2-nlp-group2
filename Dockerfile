FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]
