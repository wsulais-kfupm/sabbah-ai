import re
import pyarabic.trans
import pyarabic.araby as araby
import pandas as pd


class Preprocess:
    def __init__(self, df):
        self.stopword_txt=open('./arstopwords.txt', 'r')
        self.txt=iter(df)

    def remove_undesired_chars(self, txt):
        for ud in ["\t","\r","\a"]:
            txt=re.sub(ud, '', txt)
        txt=re.sub('\n', ' ', txt)
        return txt

    def normalize_letters(self, txt):
        txt=araby.strip_tashkeel(txt)
        txt=araby.strip_tatweel(txt)
        txt=araby.normalize_hamza(txt, method="tasheel")
        return txt

    def remove_hashtags(self, txt):
        txt=re.sub("_", ' ', txt)
        txt=re.sub('#', '', txt)
        return txt

    def remove_stopwords(self, txt):
        for stopword in self.stopword_txt:
            txt=re.sub(stopword, '', txt)
        return txt

    def remove_repeats(self, txt):
        letters = [letter for letter in araby.LETTERS]
        clean_txt=''
        txt=araby.tokenize(txt)
        prev_char=None

        for word in txt:
            prev_char=None
            letter_count=0
            for letter in word:
                if letter in letters:
                    if letter==prev_char and letter_count < 2:
                        letter_count+=1
                        clean_txt+=letter
                    elif letter!=prev_char:
                        letter_count=0
                        prev_char=letter
                        clean_txt+=letter
            clean_txt+=' '
        return clean_txt

    def preprocess(self):
        clean_txt=[]
        for s in self.txt:
            s=self.remove_hashtags(s)
            s=self.remove_stopwords(s)
            s=self.remove_repeats(s)
            s=self.normalize_letters(s)
            clean_txt.append(s)
        return clean_txt

    