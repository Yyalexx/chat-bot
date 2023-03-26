n_chunk_begin = 0     # Номер первого чанка для обработки (0 - 23)
n_chunk_end = 4       # Номер последнего чанка для обработки плюс один (1 - 24)
import pandas as pd 
from pymorphy2 import MorphAnalyzer
import pickle
from stop_words import get_stop_words
import string

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru")) # Стоп-слова
exclude = set(string.punctuation) # Знаки пунктуации

def preprocess_txt(line, morpher=morpher, sw=sw):
    """
    Функция формирования нормальной формы слов.
        Args:
            line(string): текст, 
            morpher(analyzer): класс MorphAnalyzer из библиотеки pymorphy2,
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

def get_words_set(ser):
    """
    Функция формирует множество слов текстов.
        Args:
            ser(Series): тексты, 
        Returns:
            sentences_set(array): множество слов текстов.
    """ 
    sentences_set = set()
    for line in ser.values:
        spls = set(preprocess_txt(line, morpher, sw))
        sentences_set.update(spls)
    return sentences_set

nonprod_df= pd.read_csv('nonprod_df.csv')
chunk_vol = 50000
n_chunks = len(nonprod_df)//50000 + 1
nonprod_word_set = set()
for i in range(n_chunk_begin,n_chunk_end):
    begin = i*chunk_vol
    if (i+1)*chunk_vol > len(nonprod_df):
        end = len(nonprod_df)
    else:
        end = (i+1)*chunk_vol
    print(begin,end)
    chunk_ser = nonprod_df.question[begin:end]
    chunk_word_set = get_words_set(chunk_ser)
    file_name = 'chunk_'+str(i)+'.pkl'
    with open(file_name, 'wb') as output:
        pickle.dump(chunk_word_set, output)
