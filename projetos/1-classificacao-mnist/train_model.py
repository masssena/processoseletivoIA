import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desabilita o uso da GPU, utilizando somente a CPU para treinamento

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
# Vítor Massena dos Santos
# ---------------------------------------------------------------------------

#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
from tensorflow.keras.datasets import mnist

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25)


#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_val /= 255.0
    x_test /= 255.0

    model = keras.Sequential()
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#      Bloco 1 -- Conv2D (32), BatchNormalization, MaxPooling2D
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

#      Bloco 2 -- Conv2D (64) , BatchNormalization, MaxPooling2D
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same')) 

#      Bloco 3 -- Conv2D (64), BatchNormalization, MaxPooling2D
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))  # Dropout antes da saída, para regularização
    model.add(layers.Dense(10, activation="softmax"))


#   5. Treinar com EarlyStopping monitorando a perda de validação
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    model.fit(x_train, y_train, batch_size = 64, epochs = 10, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose = 2)

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

#   6. Exibir a acurácia de validação final no terminal
    print("\n" + "=" * 50)
    print(f"Acurácia de validação final: {val_accuracy:.4f} (perda: {val_loss:.4f})")
    print(f"Acurácia de teste:           {test_accuracy:.4f} (perda: {test_loss:.4f})")
    print("=" * 50 + "\n")

#   7. Salvar o modelo treinado como "model.h5"
    model.save("model.h5")
    print("Modelo salvo como 'model.h5'")

if __name__ == "__main__":
    main()