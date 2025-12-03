# 1. Usar una imagen base ligera de Python 3.9
FROM python:3.9-slim

# 2. Evitar que Python genere archivos .pyc y permitir logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Crear directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Instalar dependencias del sistema necesarias
# (libsndfile1 es a veces requerida para procesamiento de audio avanzado)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copiar los requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar el código de la aplicación
COPY app.py .

# 7. Exponer el puerto 5000
EXPOSE 5000

# 8. Comando para iniciar la aplicación
CMD ["python", "app.py"]