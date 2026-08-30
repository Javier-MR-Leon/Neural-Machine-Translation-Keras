import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate

# Fíjate que ahora importamos download_europarl
from utils import clean_text, encode_sequences, download_europarl 

def main():
    raw_dir = "../data/raw"
    processed_dir = "../data/processed/"
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. DESCARGA AUTOMÁTICA DEL DATASET (Evita subir 400MB a GitHub)
    en_path, es_path = download_europarl(raw_dir)
    
    en_sample, es_sample = [], []
    with open(en_path, 'r', encoding='utf-8') as f_en, open(es_path, 'r', encoding='utf-8') as f_es:
        for _ in range(500000): # El limite es customizable, se recomienda no leer todo el archivo (2 millones de lineas).
            en_line = f_en.readline()
            es_line = f_es.readline()
            if not en_line or not es_line: break
            en_sample.append(en_line.strip())
            es_sample.append(es_line.strip())
            
    df = pd.DataFrame({"en": en_sample, "es": es_sample})
    
    # Limpia el texto (minúsculas y puntuación)
    df["en"] = df["en"].apply(clean_text)
    df["es"] = df["es"].apply(clean_text)
    
    # Ajusta Tokenizers
    es_en_pairs = list(zip(df["en"], df["es"]))
    es_en_array = np.array(es_en_pairs)
    
    source_tokenizer = Tokenizer()
    target_tokenizer = Tokenizer()
    source_tokenizer.fit_on_texts(es_en_array[:, 0])
    target_tokenizer.fit_on_texts(es_en_array[:, 1])
    
    # Guarda los tokenizers
    with open(os.path.join(processed_dir, 'source_tokenizer.pkl'), 'wb') as handle:
        pickle.dump(source_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(processed_dir, 'target_tokenizer.pkl'), 'wb') as handle:
        pickle.dump(target_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"\nVocabulario INGLÉS: {len(source_tokenizer.word_index) + 1}")
    print(f"\nVocabulario ESPAÑOL: {len(target_tokenizer.word_index) + 1}")
    
    # Train/Test Split y Codificación de secuencias
    train, test = train_test_split(es_en_array, test_size=0.2, random_state=42)
    
    max_source_length = 30
    max_target_length = 30
    
    trainX = encode_sequences(source_tokenizer, max_source_length, train[:, 0])
    trainY = encode_sequences(target_tokenizer, max_target_length, train[:, 1])
    testX = encode_sequences(source_tokenizer, max_source_length, test[:, 0])
    testY = encode_sequences(target_tokenizer, max_target_length, test[:, 1])
    
    trainY = trainY.reshape((trainY.shape[0], trainY.shape[1], 1))
    testY = testY.reshape((testY.shape[0], testY.shape[1], 1))
    
    # Guarda las matrices Numpy
    np.save(os.path.join(processed_dir, 'trainX.npy'), trainX)
    np.save(os.path.join(processed_dir, 'trainY.npy'), trainY)
    np.save(os.path.join(processed_dir, 'testX.npy'), testX)
    np.save(os.path.join(processed_dir, 'testY.npy'), testY)
    
    # Guardar test original para ver los textos en inferencia
    np.save(os.path.join(processed_dir, 'test_raw.npy'), test)

if __name__ == "__main__":
    main()
