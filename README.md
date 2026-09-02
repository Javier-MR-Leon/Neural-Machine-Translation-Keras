# Neural Machine Translation (EN-ES) con Seq2Seq, Atención y GloVe

Este proyecto implementa un sistema avanzado de Traducción Automática Neuronal (NMT) del Inglés al Español utilizando **TensorFlow/Keras**. El modelo traduce frases complejas gracias a una arquitectura de Redes Neuronales Recurrentes (LSTM) complementada con Mecanismos de Atención y embeddings preentrenados.

## Características Principales

* **Arquitectura Seq2Seq:** Encoder-Decoder con unidades LSTM.
* **Mecanismo de Atención:** Permite al Decoder "enfocarse" en las partes relevantes de la frase en inglés al generar cada palabra en español.
* **Embeddings Preentrenados (GloVe):** Utiliza `glove.42B.300d` de la Universidad de Stanford para transferir conocimiento semántico del inglés, acelerando el entrenamiento y mejorando la precisión.
* **Teacher Forcing:** Técnica de entrenamiento optimizada pasando las palabras objetivo desplazadas al Decoder.
* **Pipeline Automático:** Scripts integrados para la descarga y preprocesamiento automático del dataset Europarl y los pesos de GloVe.

## Estructura del Proyecto

```text
├── src/
│   ├── 01_data_prep.py      # Descarga Europarl, limpia texto y tokeniza
│   ├── 02_custom_embed.py   # Entrena el modelo Seq2Seq Base
│   ├── 03_glove_embed.py    # Entrena el modelo Seq2Seq Avanzado con GloVe
│   ├── 04_inference.py      # Traduce frases nuevas comparando ambos modelos
│   └── utils.py             # Funciones auxiliares (limpieza, descargas, decodificación)
├── data/                    # (Ignorado en Git) Datasets y matrices Numpy
├── models/                  # (Ignorado en Git) Modelos .keras guardados
├── requirements.txt         
└── README.md
```

## Instalación y Configuración

### 1. Clonar el repositorio:
```bash
git clone [https://github.com/Javier-MR-Leon/Neural-Machine-Translation-Keras.git](https://github.com/Javier-MR-Leon/Neural-Machine-Translation-Keras.git)
cd Neural-Machine-Translation-Keras
```

### 2. Entorno de Python
Se recomienda el uso de un entorno virtual (**Conda**) para garantizar la compatibilidad de las librerías.
* **Versión recomendada:** Python 3.10.19
* **Instalación de dependencias:**
```bash
pip install -r requirements.txt
```
## Uso
### Carga herramientas: limpieza, descarga de datasets y decodificación:
```bash
python src/utils.py
```

### Preparar Datos: Descarga el corpus (Europarl) y genera los tokens:
```bash
python src/01_data_prep.py
```

### Entrenar Modelo Base: Inicia el entrenamiento desde cero (Opcional):
```bash
python src/02_custom_embed.py
```

### Entrenar Modelo GloVe: Descarga los embeddings de Stanford y entrena el modelo avanzado:
```bash
python src/03_glove_embed.py
```

### Hacer Inferencias: Evalúa los modelos traduciendo frases de prueba:
```bash
python src/04_inference.py
```
