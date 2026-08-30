import os
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

def model_Seq2Seq (in_vocab_size, embedding_vec_length, max_text_length, out_vocab_size, units):
    """
    Modelo Encoder-Decoder (Seq2Seq) usando la API Funcional de Keras.
    """

    # ENCODER 
    encoder_inputs = Input(shape=(max_text_length,), name="encoder_inputs")
    enc_emb = Embedding(in_vocab_size, embedding_vec_length, mask_zero=True)(encoder_inputs)
    
    encoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    
    # Guardamos los estados finales (memoria que se pasará al decoder)
    encoder_states = [state_h, state_c]
    
    # DECODER
    decoder_inputs = Input(shape=(max_text_length,), name="decoder_inputs")
    dec_emb = Embedding(out_vocab_size, embedding_vec_length, mask_zero=True)(decoder_inputs)
    
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
  
    # CAPA DE ATENCIÓN
    context_vector = Attention(name="Capa_Atencion")([decoder_outputs, encoder_outputs])
    decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
    
    # PREDICCIÓN FINAL
    decoder_dense = Dense(out_vocab_size, activation="softmax")
    output = decoder_dense(decoder_combined)
    
    model = Model([encoder_inputs, decoder_inputs], output)
    return model
  
def main():
    processed_dir = "../data/processed/"
    models_dir = "../models/"
    os.makedirs(models_dir, exist_ok=True)
    
    # Carga Tokenizers y Datos
    with open(os.path.join(processed_dir, 'source_tokenizer.pkl'), 'rb') as handle:
        source_tokenizer = pickle.load(handle)
    with open(os.path.join(processed_dir, 'target_tokenizer.pkl'), 'rb') as handle:
        target_tokenizer = pickle.load(handle)
        
    trainX = np.load(os.path.join(processed_dir, 'trainX.npy'))
    trainY = np.load(os.path.join(processed_dir, 'trainY.npy'))
    testX = np.load(os.path.join(processed_dir, 'testX.npy'))
    testY = np.load(os.path.join(processed_dir, 'testY.npy'))
    
    source_vocab_size = len(source_tokenizer.word_index) + 1
    target_vocab_size = len(target_tokenizer.word_index) + 1
    max_length = 30
    units = 100
    embedding_vec_length = 200
    
    # Prepara datos desplazados para Teacher Forcing
    # Matrices 2D llenas de ceros para las entradas del Decoder
    decoder_input_train = np.zeros((trainY.shape[0], max_length))
    decoder_input_test = np.zeros((testY.shape[0], max_length))
    
    # Desplazamos las palabras una posición a la derecha (para que vaya siempre un paso por detrás)
    decoder_input_train[:, 1:] = trainY[:, :-1, 0]
    decoder_input_test[:, 1:] = testY[:, :-1, 0]

    # Definimos modelo Seq2Seq 
    mt_model = model_Seq2Seq(source_vocab_size, embedding_vec_length, max_length, target_vocab_size, units)
    
    rms = RMSprop(learning_rate=0.001)
    mt_model.compile(optimizer=rms, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Configurar Checkpoint
    filename = os.path.join(models_dir, "model_ta_en_es_pro.keras")
    checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    
    # Pasamos las DOS entradas en formato de lista: [Inglés, Español desplazado] - Iniciamos entrenamiento
    mt_model.fit(
        x=[trainX, decoder_input_train], 
        y=trainY, 
        epochs=50, 
        batch_size=128, 
        validation_data=([testX, decoder_input_test], testY), 
        callbacks=[checkpoint], 
        verbose=1
    )
    
    print(f"[+] Entrenamiento completado. Modelo guardado en {filename}")

if __name__ == "__main__":
    main()
