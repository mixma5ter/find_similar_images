import json
import os
import re

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as image_utils
from tqdm import tqdm

# директория поиска
SEARCH_DIR = 'Images'
# пороговое значение вероятности
THRESHOLD = .99


class Images:
    """Поиск изображений в директории."""

    def __init__(self):
        self.file = []
        self.exts = ['.png', '.jpg', '.jpeg']
        self.chk_pat = '(?:{})'.format('|'.join(self.exts))

    def parse_img(self, dir_name=os.getcwd()):
        """Запускает поиск файлов.

        :param dir_name: Директория для поиска.
        Если директория не указана, ищет в текущей директории.
        """
        path = f'{dir_name}/'
        for root, _, files in os.walk(path):
            for ix, file in enumerate(files):
                if bool(re.search(self.chk_pat, file.lower(), flags=re.I)):
                    self.file.append(os.path.join(root, file))


def deep_vector(x):
    """Преобразует многомерную матрицу изображения в вектор 1x512."""
    t_arr = image_utils.load_img(x, target_size=(224, 224))
    t_arr = image_utils.img_to_array(t_arr)
    t_arr = np.expand_dims(t_arr, axis=0)
    # предварительная обработка изображения
    processed_img = preprocess_input(t_arr, data_format=None)
    pred = model.predict(processed_img)
    return pred


def similarity(vector1, vector2):
    """Сравнивает вектора изображений. Возвращает вероятность совпадения."""
    prob = np.dot(vector1, vector2.T) / np.dot(np.linalg.norm(vector1, axis=1, keepdims=True),
                                               np.linalg.norm(vector2.T, axis=0, keepdims=True))
    return float(prob[0][0])


def find_doubles(data):
    """Находит похожие изображения в списке. Возвращает словарь."""
    result = {}
    counter = 0
    for i in range(len(data)):
        temp = [arr.file[i]]
        for j in range(i + 1, len(data)):
            P = similarity(data[i], data[j])
            if P > THRESHOLD:
                temp.append(arr.file[j])
        if len(temp) > 1:
            result[counter] = temp
            counter += 1
    return result


if __name__ == '__main__':
    arr = Images()
    arr.parse_img(SEARCH_DIR)

    # обученная сеть VGG16 на данных imagenet
    model = VGG16(include_top=False,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling='max')

    arr_list = []  # временное хранилище векторов изображения
    error_list = []  # ошибки

    # создаем вектор изображения и помещаем в список arr_list
    for i in tqdm(arr.file):
        try:
            _vector = deep_vector(i)
            arr_list.append(_vector)
        except Exception as e:
            error_list.append(i)

    # получаем похожие изображения
    doubles = find_doubles(arr_list)

    # сохраняем полученное
    with open('data.json', 'w', encoding='windows-1251') as file:
        json.dump(doubles, file, indent=4, ensure_ascii=False)

    with open('data_error.json', 'w', encoding='windows-1251') as file:
        json.dump(error_list, file, indent=4, ensure_ascii=False)

    print(f'{len(doubles)} doubles.')
    print(f'{len(error_list)} errors.')
