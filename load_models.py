import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# models
import pickle as pkl
import spacy
import gensim
import xgboost as xgb
# my files
from functions import *

class Models:
	def __init__(self):
		self.wv_model = self.load_w2v()
		self.booster = self.load_xgb()
		self.vectorizer = self.load_tfidf()
		self.nlp = self.load_spacy()

	def load_w2v(self, path='./models/wv300cian.model'):
		return gensim.models.KeyedVectors.load_word2vec_format(path)

	def load_xgb(self, path='./models/xgb.model'):
		# print(os.getcwd())
		bst = xgb.XGBClassifier()  # init model
		bst.load_model(path)
		return bst

	def load_tfidf(self, path='./models/vectorizer.model'):
		with open(path, 'rb') as f:
			vectorizer24 = pkl.load(f)
		return vectorizer24

	def load_spacy(self, path='./spacy-ru/ru2/'):
		nlp = spacy.load(path, disable=['ner', 'parser'])
		return nlp

models = Models()