
from functools import lru_cache
import pickle


@lru_cache(maxsize=1)
def get_all_glosses():
    with open("./vocab.pkl", "rb") as f:
       vocab = pickle.load(f) 
    return vocab

def translate_result(result):
    glosses = get_all_glosses()
    return [glosses[int(r)] for r in result]
    