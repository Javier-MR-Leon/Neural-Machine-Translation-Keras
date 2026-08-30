import os
import string
import zipfile
import numpy as np
import urllib.request
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    """Convierte a minúsculas y elimina signos de puntuación."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def encode_sequences(tokenizer, length, lines):
    """Codifica textos a secuencias numéricas y aplica padding."""
    seq = tokenizer.texts_to_sequences(lines)
    seq_padded = pad_sequences(seq, maxlen=length, padding='post')
    return seq_padded

def get_word(n, tokenizer):
    """Devuelve la palabra correspondiente a un índice numérico."""
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

def decode_predictions(preds, tokenizer):
    """Decodifica las secuencias numéricas de vuelta a texto, eliminando duplicados seguidos."""
    preds_text = []
    for seq in preds:
        temp = []
        for i, idx in enumerate(seq):
            word = get_word(idx, tokenizer)
            if i > 0:
                if word == get_word(seq[i-1], tokenizer) or word is None:
                    continue
                else:
                    temp.append(word)
            else:
                if word is not None:
                    temp.append(word)
        preds_text.append(' '.join(temp))
    return preds_text

def download_europarl(raw_dir):
    """
    Descarga y extrae el dataset Europarl (EN-ES) desde OPUS si no existe localmente.
    """
    en_file = os.path.join(raw_dir, "Europarl.en-es.en")
    es_file = os.path.join(raw_dir, "Europarl.en-es.es")
    
    # Comprobar si ya están descargados
    if os.path.exists(en_file) and os.path.exists(es_file):
        print("[*] Los archivos de Europarl ya existen localmente. Omitiendo descarga.")
        return en_file, es_file
    
    url = "https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/en-es.txt.zip"
    zip_path = os.path.join(raw_dir, "europarl_en_es.zip")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    # Descargar el archivo zip
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    # Limpiar el archivo zip para ahorrar espacio
    os.remove(zip_path)
    
    return en_file, es_file

def download_glove(embeddings_dir):
    """
    Descarga y extrae los embeddings GloVe 42B 300d desde la web oficial de Stanford.
    """
    glove_file = os.path.join(embeddings_dir, "glove.42B.300d.txt")
    
    if os.path.exists(glove_file):
        print("Archivo GloVe ya existe localmente. Omitiendo descarga.")
        return glove_file
    
    os.makedirs(embeddings_dir, exist_ok=True)
    url = "http://nlp.stanford.edu/data/glove.42B.300d.zip"
    zip_path = os.path.join(embeddings_dir, "glove.zip")
    
    try:
        # Descarga
        urllib.request.urlretrieve(url, zip_path)
        print("\nDescarga de GloVe completada. Extrayendo archivo (casi 5 GB, paciencia)...")
        
        # Extracción
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(embeddings_dir)
            
        # Borramos el zip 
        os.remove(zip_path)
        print("\nExtracción completada.\n")
    except Exception as e:
        print(f"\nError durante la descarga de GloVe: {e}")
        
    return glove_file
    
