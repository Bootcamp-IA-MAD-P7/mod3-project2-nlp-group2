# ---- Frontend build stage ----
FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Backend stage ----
FROM python:3.13-slim AS backend
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev --frozen
COPY . .
COPY --from=frontend /app/frontend/dist ./frontend/dist
EXPOSE 8000
CMD ["uv", "run", "python", "main.py"]