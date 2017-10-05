import happy_2 as hp
import lightgbm as lgb
import numpy as np
import pandas as pd

target = hp.train_feats['Is_Response']

# set the data in format lgb accepts
d_train = lgb.Dataset(hp.train_feats1, label = target)

## set parameters
## you can tune the parameters can try to better score

params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.05, 
    'max_depth': 7, 
    'num_leaves': 21, 
    'feature_fraction': 0.3, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5}

lgb_cv = lgb.cv(params, d_train, num_boost_round=500, nfold= 5, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=40)


## get nround value which hd lowest error
nround = lgb_cv['binary_error-mean'].index(np.min(lgb_cv['binary_error-mean']))

## train the model
model = lgb.train(params, d_train, num_boost_round=nround)

## make predictions
preds = model.predict(hp.test_feats1)

# make submission

def to_labels(x):
    if x > 0.66:  # cutoff - you can change it and see if accuracy improves or plot AUC curve. 
        return "happy"
    return "not_happy"

sub3 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds})
sub3['Is_Response'] = sub3['Is_Response'].map(lambda x: to_labels(x))
sub3 = sub3[['User_ID','Is_Response']]
sub3.to_csv('sub3_lgb.csv', index=False) # 0.85518