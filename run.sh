#!/bin/bash

echo "🚀 Iniciando entorno de Machine Learning con Tensores"
echo "================================================="

# Verificar si Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

# Verificar si docker-compose está instalado
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Por favor instala Docker Compose primero."
    exit 1
fi

# Crear directorio de notebooks si no existe
mkdir -p notebooks data

# Construir y ejecutar el contenedor
echo "🔨 Construyendo imagen Docker..."
docker-compose build

echo "🚀 Iniciando contenedor..."
docker-compose up -d

# Esperar a que Jupyter esté listo
echo "⏳ Esperando a que Jupyter Notebook esté listo..."
sleep 5

# Obtener el token de Jupyter
echo "📋 Obteniendo información de acceso..."
TOKEN=$(docker-compose logs ml-notebook 2>/dev/null | grep -o 'token=[a-z0-9]*' | tail -1)

if [ ! -z "$TOKEN" ]; then
    echo "✅ Jupyter Notebook está ejecutándose!"
    echo ""
    echo "🌐 Accede a: http://localhost:8888"
    echo "🔑 $TOKEN"
    echo ""
    echo "📓 El notebook está en: notebooks/Chapter15-Backpropagation-Tensores-Python.ipynb"
else
    echo "⚠️  No se pudo obtener el token automáticamente."
    echo "   Ejecuta: docker-compose logs ml-notebook | grep token"
fi

echo ""
echo "🛑 Para detener: docker-compose down"
echo "📊 Para ver logs: docker-compose logs -f ml-notebook"