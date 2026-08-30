import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import decode_predictions

def predict_sequence(model, source_seq, max_length):
    """
    Función recursiva para hacer inferencia en un modelo Seq2Seq con Teacher Forcing.
    Predice la frase palabra por palabra.
    """
    # Inicializamos la entrada del decoder vacía
    decoder_input = np.zeros((1, max_length))
    
    for i in range(max_length):
        prediction = model.predict([source_seq, decoder_input], verbose=0)
        
        # Obtenemos el ID de la palabra con mayor probabilidad 
        predicted_id = np.argmax(prediction[0, i, :])
        
        # Si predice 0 (padding), significa que la frase ha terminado
        if predicted_id == 0:
            break
            
        if i + 1 < max_length:
            decoder_input[0, i + 1] = predicted_id
            
    return decoder_input

def main():
    processed_dir = "../data/processed/"
    model_filename = "../models/model_ta_en_es_glove_pro.keras" 
    
    if not os.path.exists(model_filename):
        print(f"\nERROR: Modelo no encontrado en {model_filename}")
        return

    mt_model = load_model(model_filename)
    
    with open(os.path.join(processed_dir, 'target_tokenizer.pkl'), 'rb') as handle:
        target_tokenizer = pickle.load(handle)
        
    testX = np.load(os.path.join(processed_dir, 'testX.npy'))
    test_raw = np.load(os.path.join(processed_dir, 'test_raw.npy'))
    
    max_length = testX.shape[1]
    
    # Predecimos una muestra pequeña 
    sample_size = 20
    
    preds = []
    for i in range(sample_size):
        source_seq = testX[i:i+1]
        
        # Generamos la traducción
        pred_seq = predict_sequence(mt_model, source_seq, max_length)
        preds.append(pred_seq[0])
        print(f"\nTraduciendo frase {i+1}/{sample_size}...")
        
    preds_text = decode_predictions(preds, target_tokenizer)
    
    results_df = pd.DataFrame({
        "Frase origen (EN)": test_raw[:sample_size, 0],
        "Traducción real (ES)": test_raw[:sample_size, 1], 
        "Traducción generada": preds_text
    })
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nResultados Finales:\n")
    print(results_df.to_string())

if __name__ == "__main__":
    main()
