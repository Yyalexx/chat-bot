n_chunk = 11   # 0-11
chunk_vol = 100000

import pandas as pd
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import string
import numpy as np
import gensim

nonprod_df= pd.read_csv('nonprod_df.csv')

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru")) # Стоп-слова
exclude = set(string.punctuation) # Знаки пунктуации

def get_vector(line, N_vect=100):
    question = preprocess_txt(line)
    n_w2v = 0
    vector = np.zeros(N_vect)
    for word in question:
        if word in model.wv:
            vector += model.wv[word]
            n_w2v += 1
    if n_w2v > 0:
        vector = vector / n_w2v
    return vector

def get_arr(ser, N_vect=100):

    arr = np.zeros([len(ser), N_vect])
    i = 0
    for line in ser.values:
        arr[i] = get_vector(line, N_vect)
        i += 1
    return arr

def preprocess_txt(line, morpher=morpher, sw=sw):
    """
    Функция формирования нормальной формы слов.
        Args:
            line(string): текст, 
            morpher(pymorphy2.analyzer.MorphAnalyzer): класс MorphAnalyzer из библиотеки pymorphy2,
            sw(): множество стоп-слов.
        Returns:
            spls(list): список слов в нормальной форме.
    """
    spls = " ".join(i.strip() for i in line.split(',')).split('(')
    spls = " ".join(" ".join(" ".join(spls).split(')')).split('.')).split('-')
    spls = " ".join(spls).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i.replace('?', '').replace('!', '') for i in spls if i not in exclude and i not in sw and i != ""]
    return spls


model = gensim.models.Word2Vec.load("w2v_nonprod.model")

index_map = {}

# Для чанка определяем индексы обрабатываемых строк nonprod_df

begin = n_chunk * chunk_vol
end = min(len(nonprod_df), (n_chunk + 1)*chunk_vol)
print(begin,end)
arr = get_arr(nonprod_df.iloc[begin:end].question)
file_name = './arrays/nonprod_arr_'+str(n_chunk)+'.npy'
np.save(file_name, arr)  
