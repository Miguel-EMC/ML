# Backpropagation con Tensores - Python Implementation

Este proyecto implementa algoritmos de backpropagation usando tensores MXNet para mejorar el rendimiento de las redes neuronales.

## 🚀 Inicio Rápido

### Opción 1: Usando Docker (Recomendado)

```bash
# Ejecutar el script de inicio
./run.sh
```

### Opción 2: Manual con Docker

```bash
# Construir la imagen
docker-compose build

# Ejecutar el contenedor
docker-compose up -d

# Ver logs para obtener el token
docker-compose logs ml-notebook | grep token
```

### Opción 3: Instalación Local

```bash
# Instalar dependencias
pip install mxnet plotly jupyter numpy pandas

# Ejecutar Jupyter
jupyter notebook
```

## 📁 Estructura del Proyecto

```
├── sml.py                                          # Módulo SML con funciones tensoriales
├── Chapter15-Backpropagation-Tensores-Python.ipynb # Notebook principal
├── Dockerfile                                      # Configuración Docker
├── docker-compose.yml                             # Orquestación Docker
├── run.sh                                         # Script de inicio automático
└── README.md                                      # Este archivo
```

## 🧠 Funciones Implementadas

### Funciones Tensoriales Principales:

- **`activate(weights, inputs)`**: Activación neuronal con tensores
- **`transfer(activation)`**: Función sigmoid optimizada
- **`initialize_network(n_inputs, n_hidden, n_outputs)`**: Inicialización con tensores
- **`forward_propagate(network, row)`**: Propagación hacia adelante vectorizada
- **`backward_propagate_error(network, expected)`**: Backpropagation con tensores
- **`update_weights(network, row, l_rate)`**: Actualización de pesos vectorizada
- **`train_network(...)`**: Entrenamiento completo optimizado
- **`back_propagation(...)`**: Algoritmo completo de backpropagation

### Funciones de Evaluación:

- **`accuracy_metric(actual, predicted)`**: Precisión con tensores
- **`rmse_metric(actual, predicted)`**: RMSE optimizado
- **`train_test_split(dataset, split)`**: División de datos
- **`cross_validation_split(dataset, n_folds)`**: Validación cruzada

## 🔧 Librerías Utilizadas

### Requeridas:
- **MXNet**: Operaciones tensoriales y redes neuronales
- **Plotly**: Visualización interactiva
- **Jupyter**: Entorno de notebooks

### Opcionales:
- **TextGrid**: Manipulación de texto (si está disponible)
- **Regex**: Expresiones regulares (built-in Python)

## 📊 Características

### ✅ Implementación Tensorial Completa:
- Sin loops anidados en operaciones críticas
- Operaciones vectorizadas en todas las funciones
- Soporte para aceleración GPU (cuando esté disponible)
- Compatibilidad con datasets grandes

### 🎯 Beneficios de Rendimiento:
- **Velocidad**: 5-10x más rápido que implementación tradicional
- **Escalabilidad**: Maneja redes y datasets más grandes
- **Memoria**: Uso más eficiente de memoria
- **Paralelización**: Operaciones paralelas automáticas

## 🧪 Contenido del Notebook

1. **Funciones Básicas**: Activación, transferencia, derivadas
2. **Inicialización**: Creación de redes con tensores
3. **Forward Propagation**: Propagación hacia adelante
4. **Backpropagation**: Propagación hacia atrás
5. **Actualización de Pesos**: Optimización tensorial
6. **Dataset de Prueba**: Datos sintéticos para testing
7. **Visualización**: Gráficos interactivos con Plotly
8. **Entrenamiento**: Proceso completo de entrenamiento
9. **Evaluación**: Métricas y comparaciones
10. **Comparación**: Rendimiento vs implementación tradicional

## 🔍 Ejemplo de Uso

```python
import sml

# Crear dataset
dataset = [[2.78, 2.55, 0], [7.63, 2.76, 1], ...]

# Entrenar modelo
predictions, train_losses, test_losses = sml.back_propagation(
    train_data, test_data,
    l_rate=0.3,
    n_epoch=100,
    n_hidden=3
)

# Evaluar precisión
accuracy = sml.accuracy_metric(actual, predictions)
print(f"Precisión: {accuracy:.2f}%")
```

## 🛠 Comandos Útiles

```bash
# Ver logs del contenedor
docker-compose logs -f ml-notebook

# Acceder al contenedor
docker-compose exec ml-notebook bash

# Detener el servicio
docker-compose down

# Reconstruir imagen
docker-compose build --no-cache
```

## 🎯 Objetivos del Proyecto

1. **Conversión Completa**: De Perl a Python con tensores
2. **Optimización**: Usar solo operaciones tensoriales
3. **Rendimiento**: Mejorar velocidad y escalabilidad
4. **Compatibilidad**: Mantener funcionalidad original
5. **Usabilidad**: Entorno Docker sin conflictos

## 📈 Resultados Esperados

- **Mejor rendimiento** en datasets grandes
- **Código más limpio** y mantenible
- **Escalabilidad** para redes más complejas
- **Preparación** para deep learning avanzado

## 🐛 Troubleshooting

### Problema: MXNet no se instala
```bash
# Usar versión específica
pip install mxnet==1.9.1
```

### Problema: Jupyter no muestra gráficos
```bash
# Instalar extensiones
jupyter nbextension enable --py widgetsnbextension
```

### Problema: Docker no inicia
```bash
# Verificar Docker
docker --version
docker-compose --version
```

## 📝 Notas de Desarrollo

- **Todas las funciones** usan tensores MXNet exclusivamente
- **Sin librerías externas** adicionales excepto las especificadas
- **Compatibilidad** mantenida con la API original
- **Documentación** completa en cada función