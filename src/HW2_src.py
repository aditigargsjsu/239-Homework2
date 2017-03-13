
import sys
sys.path.append('')

import warnings
warnings.filterwarnings("ignore")

import operator
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import numpy
import math
import datetime

#print 'Staring test now!'
#print datetime.datetime.now()

stop_words = get_stop_words('english')

def remove_stopwords(raw_string):
    raw_string = raw_string.lower()
    d = raw_string.split()
    new_list=[]
    for i in range(0, len(d)):
        if d[i] not in stop_words:
            new_list.append(d[i])
    filtered_string = ' '.join(new_list)
    return filtered_string

def get_cosine_similarity(array1,array2):
    dot_product = numpy.dot(array1,array2)
    mod_array1 = numpy.linalg.norm(array1)
    mod_array2 = numpy.linalg.norm(array2)
    cos_similarity = dot_product/(mod_array1*mod_array2)
    return cos_similarity

# -------------------- #
# FILTERING AND MOVING TEST AND TRAIN DATA IN A LIST. AND SAVING ALL THEIR POLARITIES IN A LIST

f1 = open("/Users/Downloads/train.dat","r")
x = f1.readlines()
filtered_train_data_list = []
for i in range(0, len(x)):
    filtered_review = remove_stopwords(x[i])
    filtered_train_data_list.append(filtered_review)
f1.close()

f3 = open("/Users/Downloads/test.dat","r")
y = f3.readlines()
filtered_test_data_list = []
for i in range(0, len(y)):
    filtered_review = remove_stopwords(y[i])
    filtered_test_data_list.append(filtered_review)
f3.close()

total_polarities_set = []
with open ("/Users/Downloads/train.dat","r") as f:
    for line in f:
        total_polarities_set.append(line[0])


# NOW, TEST AND TRAIN DATA IS FILTERED AND ADDED IN A LIST
# ----------------- #



# -------- NOW USING A VOCAB LIST TO CONVERT REVIEWS INTO A VECTOR ----- #

vocab = ['good','great','life saver','help','helped','simple','durable','easy','glad','painful','enjoy','like','love','hate','bad','hard','ugly','safe','risky','difficult','best','loved','hated','returned','return','perfect','not good','uncomfortable','impressed','waste','terrible','worthy','worthless','comfortable','useless','adore','complain','attractive','beautiful','wonderful','disappointed','recommend','horrible','disgusting','bulky']
cv = CountVectorizer(vocabulary=vocab)
#cv = CountVectorizer(max_features = 23400)

test_data_vector = cv.fit_transform(filtered_test_data_list).toarray()
train_data_vector = cv.fit_transform(filtered_train_data_list).toarray()

# Also adding TFIDF-Transformer now
#tfidf_transformer = TfidfTransformer()
#X_test_tfidf = tfidf_transformer.fit_transform(test_data_vector)
#X_train_tfidf = tfidf_transformer.fit_transform(train_data_vector)


for a in range (0,len(test_data_vector)):
    cosine_similarity_dict={}
    
    for i in range (0,len(train_data_vector)):
        cs = get_cosine_similarity(test_data_vector[a],train_data_vector[i]).tolist()
        if math.isnan(cs)==True:
            #print 'is nan detected. making the similarity 0'
            cs = float(0)
        cosine_similarity_dict[i]=cs

    sorted_cosine_similarity_list = sorted(cosine_similarity_dict.items(), key=operator.itemgetter(1))
    sorted_cosine_similarity_list.reverse()
    #print 'reversed cosine similarity list is'
    #print sorted_cosine_similarity_list


    k=5
    top_k_neighbors = sorted_cosine_similarity_list[:k]
    #print top_k_neighbors

    # ----- #

    # ---- TO USE EUCLIDEAN DISTANCE ---- #
    ##euclidean_distance_dict={}
    ##
    ##for i in range (0,len(train_data_vector)):
    ##	x = euclidean(test_data_vector[0],train_data_vector[i]).tolist()
    ##	euclidean_distance_dict[i]=x
    ##
    ##sorted_euclidean_distance_list = sorted(euclidean_distance_dict.items(), key=operator.itemgetter(1))
    ##
    ##k=5
    ##top_k_neighbors = sorted_euclidean_distance_list[:k]

    # --------- #

    polarity_of_top_neighbors = []
    for j in range(0,k):
        neighbor_number = top_k_neighbors[j][0]
        neighbor_polarity = total_polarities_set[neighbor_number]
        polarity_of_top_neighbors.append(neighbor_polarity)

    number_of_plus = polarity_of_top_neighbors.count('+')
    number_of_minus = polarity_of_top_neighbors.count('-')

    if number_of_plus > number_of_minus:
        classified_polarity_of_test_review='+1'
    else:
        classified_polarity_of_test_review='-1'

    f4 = open("/Users/aditi/Downloads/answers_k5.dat","a+")
    f4.write(classified_polarity_of_test_review+"\n")
    f4.close()

print 'Classification of test reviews complete!'
#print datetime.datetime.now()
