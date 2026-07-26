# Processo Seletivo – Intensivo Maker | AI | Relatório do Candidato

### Vítor Massena dos Santos
### github.com/masssena

## Projeto 1 - Classificação MNIST

Este projeto foi desenvolvido em python a partir do banco MNIST e da plataforma TensorFlow, construindo e treinando o modelo para identificar dígitos numerais de 0 a 9 escritos à mão para, por fim, otimizá-lo para dispositivos Edge.

O modelo foi construído e treinado em *train_model.py*. Inicialmente, o dataset MNIST é carregado e as imagens são normalizadas para o padrão exigido, que serão utilizadas para o treinamento do modelo. Em seguida, é construído o modelo com três blocos convolucionais, utilizando Conv2D (filtros de extração de informações relevantes, com um grid 3x3, o primeiro com 32 e os demais com 64 filtros), BatchNormalization (padronizando as saídas) e MaxPooling2D (filtro de redução de imagens e mapas de características). 

Foi implementado, como requerido, a técnica de EarlyStopping, encerrando o treinamento quando o desempenho para de melhorar, aguardando três épocas antes de finalizar a execução.


### Bibliotecas utilizadas

Foram utilizadas as bibliotecas:

- TensorFlow (2.21)
- Keras (3.12.3)
- Scikit Learn (1.7.2)
- Numpy (2.4.6)
- Módulo OS

### Técnica de otimização 

O modelo foi otimizado com **Dynamic Range Quantization** após ser convertida para *.tflite* (TensorFlow Lite). Essa técnica foi escolhida pela sua facilidade de implementação e sua eficiência para sistemas Edge (acelerando a velocidade de inferência) sem necessidade de retreinamento.

### Resultados Obtidos


O que é perceptivelmente um alto valor de acurácia, demonstrando a qualidade do modelo.

Ao construir o modelo *model.h5*, foram verificados os seguintes valores de acurácia:


___
Acurácia de validação final: 0.9899 (perda: 0.0448)

Acurácia de teste:           0.9910 (perda: 0.0319)
___


Assim, para finalizar, a execução do exemplo de inferência no modelo otimizado com 10 amostras de teste possuiu o seguinte resultado:

___
Rodando inferencia em 10 amostras usando model.tflite:

Amostra 1: predito=7 | real=7

Amostra 2: predito=2 | real=2

Amostra 3: predito=1 | real=1

Amostra 4: predito=0 | real=0

Amostra 5: predito=4 | real=4

Amostra 6: predito=1 | real=1

Amostra 7: predito=4 | real=4

Amostra 8: predito=9 | real=9

Amostra 9: predito=5 | real=5

Amostra 10: predito=9 | real=9
___


Este e todos os outros exemplos de testes de inferência executados possuíram resultados exatos e com nenhuma discrepância ou até mesmo erro nas predições, o que confirma os dados de acurácia de 0.9899 para validação final e 0.9910 para testes.
