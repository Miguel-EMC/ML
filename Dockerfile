# Dockerfile for MXNet + Jupyter environment
FROM ubuntu:20.04

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /workspace

# Actualizar pip
RUN python3 -m pip install --upgrade pip

# Instalar librerías específicas requeridas
RUN pip3 install \
    mxnet==1.9.1 \
    numpy==1.23.5 \
    pandas \
    plotly==5.17.0 \
    jupyter \
    notebook \
    ipywidgets

# Instalar TextGrid si está disponible
RUN pip3 install textgrid || echo "TextGrid no disponible, usando implementación propia"

# Crear directorio para notebooks
RUN mkdir -p /workspace/notebooks

# Exponer puerto de Jupyter
EXPOSE 8888

# Copiar archivos al contenedor
COPY sml.py /workspace/
COPY Chapter15-Backpropagation-Tensores-Python.ipynb /workspace/notebooks/

# Comando por defecto
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
