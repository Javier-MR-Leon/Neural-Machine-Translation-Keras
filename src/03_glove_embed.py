import os
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import download_glove

def define_model_attention_with_glove(embedding_matrix, max_length, out_vocab_size, units=256):
    """
    Modelo Seq2Seq Avanzado con Mecanismo de Atención y pesos GloVe congelados.
    """
    vocab_size, embedding_dim = embedding_matrix.shape
    
    # ENCODER
    encoder_inputs = Input(shape=(max_length,), name="Entrada_Ingles")
    # Cargamos GloVe y congelamos pesos (trainable=False)
    enc_embed = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], 
                          mask_zero=True, trainable=False, name="GloVe_Ingles")(encoder_inputs)
    
    encoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="LSTM_Encoder")
    encoder_outputs, state_h, state_c = encoder_lstm(enc_embed)
    
    # DECODER
    decoder_inputs = Input(shape=(max_length,), name="Entrada_Espanol_Desplazada")
    dec_embed = Embedding(out_vocab_size, embedding_dim, mask_zero=True, name="Embed_Espanol")(decoder_inputs)
    
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name="LSTM_Decoder")
    decoder_outputs, _, _ = decoder_lstm(dec_embed, initial_state=[state_h, state_c])
    
    # ATENCIÓN
    context_vector = Attention(name="Capa_Atencion")([decoder_outputs, encoder_outputs])
    decoder_combined = Concatenate(axis=-1, name="Concatenacion_Atencion")([decoder_outputs, context_vector])
    
    # SALIDA
    decoder_dense = Dense(out_vocab_size, activation="softmax", name="Capa_Salida")
    output = decoder_dense(decoder_combined)
    
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output, name="Traductor_GloVe_Atencion")
    return model

def main():
    # ENTRENAMIENTO: GLOVE + ATENCIÓN 
    processed_dir = "../data/processed/"
    embeddings_dir = "../data/embeddings/"
    models_dir = "../models/"
    os.makedirs(models_dir, exist_ok=True)
    
    glove_path = download_glove(embeddings_dir)
    
    if not glove_path or not os.path.exists(glove_path):
        print("\nNo se pudo encontrar ni descargar el archivo GloVe.")
        return
        
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
    
    max_length = trainX.shape[1] 

    embeddings_index = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except Exception as e:
        print(f"Error leyendo GloVe: {e}")
        return

    # Matriz GloVe
    embedding_dim = 300
    embedding_matrix = np.zeros((source_vocab_size, embedding_dim))
    for word, i in source_tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    decoder_input_train = np.zeros((trainY.shape[0], max_length))
    decoder_input_test = np.zeros((testY.shape[0], max_length))
    decoder_input_train[:, 1:] = trainY[:, :-1, 0]
    decoder_input_test[:, 1:] = testY[:, :-1, 0]

    mt_model_glove = define_model_attention_with_glove(embedding_matrix, max_length, target_vocab_size, units=256)
    mt_model_glove.compile(optimizer=RMSprop(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    filename = os.path.join(models_dir, "model_ta_en_es_glove_pro.keras")
    checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    
    print("[*] Iniciando entrenamiento...")
    mt_model_glove.fit(
        x=[trainX, decoder_input_train], 
        y=trainY, 
        epochs=20, 
        batch_size=128, 
        validation_data=([testX, decoder_input_test], testY), 
        callbacks=[checkpoint], 
        verbose=1
    )

if __name__ == "__main__":
    main()
