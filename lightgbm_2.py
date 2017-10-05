# set data format
import happy_2 as hp
import numpy as np
d_train = lgb.Dataset(hp.train_feats2, label = target)
params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.05, 
    'max_depth': 5, 
    'num_leaves': 11,
    'feature_fraction': 0.3, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5}

## do cross validation to find nround i.e. at this round (iteration) we can expect lowest error
lgb_cv = lgb.cv(params, d_train, num_boost_round=500, nfold= 5, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=40)

nround = lgb_cv['binary_error-mean'].index(np.min(lgb_cv['binary_error-mean']))
# train model
model = lgb.train(params, d_train, num_boost_round=nround)
# make prediction
preds = model.predict(hp.test_feats2)

def to_labels(x):
    if x > 0.66:
        return "happy"
    return "not_happy"

sub4 = pd.DataFrame({'User_ID':hp.test.User_ID, 'Is_Response':preds})
sub4['Is_Response'] = sub4['Is_Response'].map(lambda x: to_labels(x))
sub4 = sub4[['User_ID','Is_Response']]
sub4.to_csv('sub4_lgb.csv', index=False)
