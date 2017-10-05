# Load Libraries
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
import HTMLParser



## join data
test['Is_Response'] = np.nan
alldata = pd.concat([train, test]).reset_index(drop=True)





def remove_url(data1):
	data2 = data2 = unicode(data1,'utf-8')
	html_parser = HTMLParser.HTMLParser()
	data2 = html_parser.unescape(data2)
	return data2

for i in range(len(alldata)):
	alldata.iloc[i,1] = remove_url(alldata.iloc[i,1])
	  # iloc acceses the cell of that column 






'''def standarize(data2):
data2 = data2.decode("utf8").encode('ascii','ignore')
for i in range(len(alldata)):
	alldata.iloc[i,1] = standarize(alldata.iloc[i,1])'''





import contractions as ct

def contract(data1):
	if data1 is not None:
		data1 = data1.split()
		data1= [ct.contractions_list[word] if word in ct.contractions_list else word for word in data1]
		data1 = " ".join(data1)
	return data1

# Remember that Null cant be operated with .split() therefore add a special case for it 

for i in range(len(alldata)):
	alldata.iloc[i,1] = contract(alldata.iloc[i,1])





# Now removing the stop words 

stops = set(stopwords.words("english"))
def remove_stop(data1):
	if data1 is not None:
		data1 = " ".join([w for w in data1.split() if w not in stops])
	return data1

for i in range(len(alldata)):
	alldata.iloc[i,1] = remove_stop(alldata.iloc[i,1])



# removing those words which do not have either A-Z or a-z or 0-9 i.e not the @ ,"<? / @ # $^&* types

def clean_again(data1):
	if data1 is not None:
		data1 = re.sub(r'[^A-Za-z0-9\s]',r'',data1)
	return data1

for i in range(len(alldata)):
	alldata.iloc[i,1] = clean_again(alldata.iloc[i,1])


# convert to lower case :


# Split attatched words like IamBoy

'''def split_attatched(data1):
	if data1 is not None:
		data1 = " ".join(re.findall('[A-Z][^A-Z]*', data1))
	return data1

for i in range(len(alldata)):
	alldata.iloc[i,1] = split_attatched(alldata.iloc[i,1])'''



# Now remove the extra spaces

def rem_space(data1):
	data1 = re.sub(r'\n',r' ',data1)
	return data1

for i in range(len(alldata)):
	alldata.iloc[i,1] = rem_space(alldata.iloc[i,1]) 



# tolower case 
def to_lower(data1):
	if data1 is not None:
		data1 = data1.lower()
	return data1

for i in range(len(alldata)):
	alldata.iloc[i,1] = to_lower(alldata.iloc[i,1])


# standarize the words like happyyyyyy to happy
import itertools
def standarize(data1):
	data1 = ''.join(''.join(s)[:2] for _, s in itertools.groupby(data1))
	return data1 


for i in range(len(alldata)):
	alldata.iloc[i,1] = standarize(alldata.iloc[i,1]) 

 

# now comes the spell checking part    go to this link    https://github.com/mattalcock/blog/blob/master/2012/12/5/python-spell-checker.rst
import spell as sp
def spell_check(data1):
	for s in data1:
		s = sp.correction(s)
	return data1




''' 
	LabelEncoder of each data ? what is this LabelEncoder? LabelEncoder labels each of the 
	unique entitites from 1 to n_labels-1,
	 for example you have a column   browser_used = ['chrome','mozilla','explorer']
	LabelEncoder will label this list by the .fit() method and thre labels namely, [0,1,2]
	will be formed and all the cells of corresponding columns of browser_used will belabeled according 
	to this labels.	
'''

cols = ['Browser_Used','Device_Used']
from sklearn import preprocessing

for x in cols:
    lbl = LabelEncoder()
    alldata[x] = lbl.fit_transform(alldata[x]) 



# Now we are going to do the thing called as CountVectrizer .
'''
 What it does ? it actually counts the number of all the words i.e the frequency 
of each word and converts it to an array of frequencies of each word 
'''

'''
These are some parameters of Countvector 
which must be kept in mind before  moving further :

analyzer : string, {'word', 'char', 'char_wb'} or callable

    Whether the feature should be made of word or character n-grams. 
    Option 'char_wb' creates character n-grams only from text inside word boundaries;
     n-grams at the edges of words are padded with space.

    If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.

preprocessor : callable or None (default)

    Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.

tokenizer : callable or None (default)

    Override the string tokenization step while preserving the preprocessing and n-grams generation steps. Only applies if analyzer == 'word'.

ngram_range : tuple (min_n, max_n)

    The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
    All values of n such that min_n <= n <= max_n will be used.

stop_words : string {'english'}, list, or None (default)

    If 'english', a built-in stop word list for English is used.

    If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
     Only applies if analyzer == 'word'.

    If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop
     words based on intra corpus document frequency of terms.

lowercase : boolean, True by default

    Convert all characters to lowercase before tokenizing.

token_pattern : string

    Regular expression denoting what constitutes a "token", only used if analyzer == 'word'. 
    The default regexp select tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as
     a token separator).

max_df : float in range [0.0, 1.0] or int, default=1.0

    When building the vocabulary ignore terms that have a document frequency strictly higher than the given 4
    threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts.
     This parameter is ignored if vocabulary is not None.

min_df : float in range [0.0, 1.0] or int, default=1

    When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. 
    This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

max_features : int or None, default=None

    If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

    This parameter is ignored if vocabulary is not None.

get_feature_names()[source]

    Array mapping from feature integer indices to feature name

X_inv : list of arrays, len = n_samples

    List of arrays of terms.

ngram_range : tuple (min_n, max_n)

    The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
    All values of n such that min_n <= n <= max_n will be used.



'''

for i in range(len(alldata)):
	alldata.iloc[i,1] = remove_stop(alldata.iloc[i,1])


countvec = CountVectorizer(analyzer='word', ngram_range = (1,1), min_df=150, max_features=600)
tfidfvec = TfidfVectorizer(analyzer='word', ngram_range = (1,1), min_df = 150, max_features=600)

'''What is TfidfVectorizer?  actually it ismore or less simillar to Countvector. what CountVectorizer dors is that it returns
	the number or frequency of each word in absolute terms but what TfidfVectorizer i.e term frequency inverse distribution frequency 
	does is that it takes into account that some words are repeated excess number of time like 'hey','hi' etc which can dominate in feature 
	engineering. therefore it neglects those words 

	What is TF-IDF?

TF-IDF stands for "Term Frequency, Inverse Document Frequency". It is a way to score the importance of words (or "terms") in 
a document based on how frequently they appear across multiple documents.

Intuitively...

    If a word appears frequently in a document, it's important. Give the word a high score.
    But if a word appears in many documents, it's not a unique identifier. Give the word a low score.

'''


# creating  features
bagofwords = countvec.fit_transform(alldata['Description'])
tfidfdata = tfidfvec.fit_transform(alldata['Description'])

# to see the bagofwords type:
countvec.get_feature_names()
tfidfvec.get_feature_names()
# to see the array wise representation
bagofwords.toarray()


# create dataframe for features
'''
	what the todense of scippy does is that returns a dense matrix representation of 
	passed matrix
	and .DataFrame(matrix) froms a dataframe of the provided matrix
	we are acrually creating a dataframe pf features so that we can pass 
	them to our classification models  
'''

bow_df = pd.DataFrame(bagofwords.todense())
tfidf_df = pd.DataFrame(tfidfdata.todense())


'''
	Now for our own convinience we name our columns as col123 format
'''
# set column names
bow_df.columns = ['col'+ str(x) for x in bow_df.columns]
tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]

# Now we need to seperate the test data and the train data from both the newly formed data get_feature_names

# create separate data frame for bag of words and tf-idf

bow_df_train = bow_df[:len(train)]
bow_df_test = bow_df[len(train):]

tfid_df_train = tfidf_df[:len(train)]
tfid_df_test = tfidf_df[len(train):]

# split the merged data file into train and test respectively
train_feats = alldata[~pd.isnull(alldata.Is_Response)]
test_feats = alldata[pd.isnull(alldata.Is_Response)]


### set target variable

train_feats['Is_Response'] = [1 if x == 'happy' else 0 for x in train_feats['Is_Response']]

# merge count (bag of word) features into train
'''
	train_feats has 5 columns while bow_df has 600 column where all these 
	600 columns are features  therefor we concat both in one train_feats1
	where cols = ['Device_Used' 'Browser_Used']
	length of train_feats1 is 38932 same as length of train

	simillarly we do with test_feats1
	length of test_feats is 29404
'''
train_feats1 = pd.concat([train_feats[cols], bow_df_train], axis = 1)
test_feats1 = pd.concat([test_feats[cols], bow_df_test], axis=1)

test_feats1.reset_index(drop=True, inplace=True)

# merge into a new data frame with tf-idf features
train_feats2 = pd.concat([train_feats[cols], tfid_df_train], axis=1)
test_feats2 = pd.concat([test_feats[cols], tfid_df_test], axis=1)


