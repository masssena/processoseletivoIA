import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
# Vítor Massena dos Santos
# ---------------------------------------------------------------------------


#   Função para comparar o tamanho dos modelos
def print_size_comparison(h5_path: str, tflite_path: str):
    h5_size_kb = os.path.getsize(h5_path) / 1024
    tflite_size_kb = os.path.getsize(tflite_path) / 1024
    reduction = (1 - tflite_size_kb / h5_size_kb) * 100
 
    print("Comparação de tamanho dos modelos:")
    print(f"  {h5_path:<20s}: {h5_size_kb:8.1f} KB")
    print(f"  {tflite_path:<20s}: {tflite_size_kb:8.1f} KB")
    print(f"  Redução de tamanho: {reduction:.1f}%")


#   1. Carregar o modelo treinado em "model.h5"
model = tf.keras.models.load_model("model.h5")

#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#   3. Aplicar uma técnica de otimização (utilizando Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

#   4. Salvar o resultado como "model.tflite"
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print_size_comparison("model.h5", "model.tflite")