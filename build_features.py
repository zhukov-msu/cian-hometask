# from load_models import *
from functions import *
# text & geo
import re
import Levenshtein
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import word_tokenize
# import gensim
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from geopy import distance
from scipy.spatial.distance import cosine
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def build_geo_features(data):
	'''
		'geo_addr_lev1', 
	    'geo_addr_lev2', 
	    'geo_addr_percent_intersect',
	    'geo_addr_fw',
	    'geo_distances'
	'''

	def get_geo_addr_lev1(userInput1, userInput2):
		return [Levenshtein.distance(a,b) for a,b in zip(userInput1, userInput2)]

	def get_geo_addr_lev2(addr1, addr2):
		return [Levenshtein.distance(''.join(a), ''.join(b)) for a,b in zip(addr1, addr2)]

	def get_geo_addr_fw(userInput1, userInput2):
		return [fuzz.ratio(a,b)/100 for a,b in zip(userInput1, userInput2)]

	def get_geo_addr_percent_intersect(addr1, addr2):
		return [len(np.intersect1d(a, b)) / max(len(a), len(b)) for a,b in zip(addr1, addr2)]

	def get_geo_distances(coord1, coord2):
		return [distance.geodesic(a,b).km for a,b in zip(coord1, coord2)]

	# geo_x = [json.loads(elem) for elem in data]

	geo_x = [literal_eval(elem['geo_x']) for elem in data]
	geo_y = [literal_eval(elem['geo_y']) for elem in data]

	coord_x = [(elem['coordinates']['lat'], elem['coordinates']['lng']) for elem in geo_x]
	coord_y = [(elem['coordinates']['lat'], elem['coordinates']['lng']) for elem in geo_y]

	userInput_x = [elem['userInput'].replace('Россия, ', '').strip() for elem in geo_x]
	userInput_y = [elem['userInput'].replace('Россия, ', '').strip() for elem in geo_y]

	addr_x = [elem['address'] for elem in geo_x]
	addr_y = [elem['address'] for elem in geo_y]

	namelist_x = [[x['name'] for x in elem] for elem in addr_x]
	namelist_y = [[x['name'] for x in elem] for elem in addr_y]

	geo_addr_lev1 = get_geo_addr_lev1(userInput_x, userInput_y)
	geo_addr_lev2 = get_geo_addr_lev2(namelist_x, namelist_y)
	geo_addr_fw = get_geo_addr_fw(userInput_x, userInput_y)
	geo_addr_percent_intersect = get_geo_addr_percent_intersect(namelist_x, namelist_y)
	geo_distances = get_geo_distances(coord_x, coord_y)

	df_feats_geo_all = pd.DataFrame(
	    np.array([geo_addr_lev1, geo_addr_lev2, geo_addr_percent_intersect, geo_addr_fw, geo_distances]).T,
	    columns=["geo_addr_lev1", "geo_addr_lev2", "geo_addr_percent_intersect", "geo_addr_fw", "geo_distances"]
	)

	return df_feats_geo_all


def build_text_features(data, models):
	def get_len(txt1, txt2):
		return [min(a,b)/max(a,b) for a,b in zip([len(x) for x in txt1], [len(x) for x in txt2])]
		
	def get_eng_perc(txt1, txt2):
		fn_eng = lambda x: np.mean([1 if l in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' else 0 for l in x])
		return [1 if max(a,b)==0 else min(a,b)/max(a,b) for a,b in zip([fn_eng(x) for x in txt1], [fn_eng(x) for x in txt2])]

	def get_w2v_similarity(txt1, txt2, w2vmodel=models.wv_model):
		vectors = get_w2v_vectors(
			txt1,
			txt2,
			w2vmodel
		)
		return cosine_similarity_my(*vectors)

	def get_tfidf_similarity(txt1, txt2, vectorizer=models.vectorizer):
		vec1 = vectorizer.transform(txt1)
		vec2 = vectorizer.transform(txt2)
		cosine_tfidf_ng = 1 - cosine_similarity(vec1,vec2).diagonal()
		return cosine_tfidf_ng

	def prepare_texts(texts, nlp=models.nlp):
		cleaning = lambda doc: [token.lemma_ for token in doc if not token.is_stop]
		brief_cleaning = [re.sub("[^A-Za-zА-Яа-я']+", ' ', str(row)).lower() for row in texts]
		txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
		txt_processed = [' '.join(token) for token in txt]
		return txt_processed
	
	txt_x = [elem['description_x'] for elem in data]
	txt_y = [elem['description_y'] for elem in data]

	txt_x_processed = prepare_texts(txt_x)
	txt_y_processed = prepare_texts(txt_y)

	text_len = get_len(txt_x, txt_y)
	eng_perc = get_eng_perc(txt_x, txt_y)
	cosine_tfidf_ng = get_tfidf_similarity(txt_x_processed, txt_y_processed)
	txt_similarity_feature = get_w2v_similarity(txt_x_processed, txt_y_processed)

	# concat text
	df_feats_text_all = pd.DataFrame(
	    np.array([text_len, eng_perc,  txt_similarity_feature,  cosine_tfidf_ng]).T, 
	    columns=["text_len", "eng_perc", 'txt_similarity_feature',  "cosine_tfidf_ng"]
	)
	return df_feats_text_all



def build_other_features(data):
	#"total_area_eq", 'category_eq', "rcnt_eq"
	area = [(row['totalarea_x'], row['totalarea_y']) for row in data]
	total_area_eq = [1 if abs(np.nan_to_num(a)-np.nan_to_num(b))<=1 else 0 for a,b in area]

	rcnt = [(row['roomscount_x'], row['roomscount_y']) for row in data]
	rcnt_eq = [1 if np.nan_to_num(a)==np.nan_to_num(b) else 0 for a,b in rcnt]

	bargainterms_x = [literal_eval(row['bargainterms_x']) for row in data]
	bargainterms_y = [literal_eval(row['bargainterms_y']) for row in data]

	# deposits = [(bargainterms_x['deposit'], bargainterms_y['deposit']) for row in data]
	# deposit_rate = [1 if max(np.nan_to_num(a),np.nan_to_num(b))==0 else min(np.nan_to_num(a),np.nan_to_num(b))/max(np.nan_to_num(a),np.nan_to_num(b)) for a,b in deposits]

	price_rate = [1 if max(a,b)==0 else min(a,b)/max(a,b) for a,b in zip([x['price'] for x in bargainterms_x], [x['price'] for x in bargainterms_y])]

	df_feats_other_all = pd.DataFrame(
    np.array([total_area_eq, rcnt_eq, price_rate]).T, 
    columns=["total_area_eq", "rcnt_eq", "price_rate"]
	)

	return df_feats_other_all

def build_features(data, models):
	df_all = pd.concat([build_geo_features(data), build_text_features(data, models), build_other_features(data)], axis=1)
	df_all = df_all[['geo_addr_lev1','geo_addr_lev2', 'geo_addr_percent_intersect','geo_addr_fw','geo_distances','text_len','eng_perc','txt_similarity_feature','cosine_tfidf_ng','price_rate','total_area_eq','rcnt_eq']]
	return df_all

