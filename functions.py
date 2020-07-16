from scipy.spatial.distance import cosine
import numpy as np

def sent2vec(s, w2vmodel):
    
    featureVec = np.zeros((300,), dtype="float32")
    nwords = 0
                               
    for w in s.split(' '):                                   
        try:                                       
            nwords = nwords + 1                                       
            featureVec = np.add(featureVec, w2vmodel[w])
        except:                                       
            continue                               
        # averaging                               
        if nwords > 0:                                   
            featureVec = np.divide(featureVec, nwords)
    return featureVec

def get_w2v_vectors(list_text1, list_text2, w2vmodel): 
    '''
    Computing the word2vec vector representation of list of sentences
    @param list_text1 : first list of sentences
    @param list_text2 : second list of sentences 
    '''
    print("Computing first vectors…")
    text1_vectors = np.zeros((len(list_text1), 300))
    for i, q in enumerate(list_text1):
        text1_vectors[i, :] = sent2vec(q, w2vmodel)
    text2_vectors = np.zeros((len(list_text2), 300))
    for i, q in enumerate(list_text2):
        text2_vectors[i, :] = sent2vec(q, w2vmodel)
    return text1_vectors, text2_vectors

def cosine_similarity_my(list_vec1, list_vec2):
    '''
    Computing the cosine similarity between two vector representation
    @param  list_text1  :   first list of sentences
    @param  list_text2  :   second list of sentences 
    '''
    cosine_dist = [cosine(x, y) for (x, y) in zip(np.nan_to_num(list_vec1), np.nan_to_num(list_vec2))]
    cosine_sim = [(1 - dist) for dist in cosine_dist]
    return cosine_sim