
import keras as K
from keras import Model
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
from random import randint

###Нейросеть обучена в GoogleColab(https://colab.research.google.com/drive/1xz0hZ-4OBDh5xjCTKNTai0Jx4tjwbEEh?usp=sharing)


###INIT
# загрузка датасета
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Нормализация
x_train = x_train / 255
x_test = x_test / 255

# Преобразование двумерного массива - изображения в трёхмерный для CNN
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Загрузка предобученной модели
encoder = K.models.load_model('./encoder.keras')
decoder = K.models.load_model('./decoder.keras')

example_images = x_test[:5000]
example_labels = y_test[:5000]

# Найдём вектора, которыми были закодированы изображения
embeddings = encoder.predict(example_images)

# Раскрасим точки в соответствии с их метками
vect = [[] for _ in range(10)]

for i in range(len(example_labels)):
  vect[example_labels[i]].append([float(embeddings[i][0]), float(embeddings[i][1])])

"""# **Генерируем новые изображения**

# **Генерируем новые изображения**
"""

###GENERATION
def gen_num(num: str, vector, model: Model):
    sample = []
    for i in num:
        i = int(i)
        x = list(vector[i][randint(0, len(vector[i]))])
        sample.append(x)
    sample = np.array(sample)
    return model.predict(sample)