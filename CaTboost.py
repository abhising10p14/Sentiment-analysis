## import library
import happy_2 as hp
from catboost import CatBoostClassifier,cv, Pool
## catboost accepts categorical columns as a list of column numbers. In this data, all columns are categorical
cat_cols = [x for x in range(502)] ## 502 == train_feats1.shape[1]
target = hp.train_feats['Is_Response']
## set parameters
## you can refer the parameters here: https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/#python-reference_parameters-list
param = {
    'use_best_model':True,
    'loss_function':'CrossEntropy',
    'eval_metric':'Accuracy',
    'iterations':1000,
    'depth':6,
    'learning_rate':0.03,
    'rsm':0.3,
    'random_seed':2017,
    
    
}

## for doing cross validation, set data in Pool format
my_dt =  Pool(hp.train_feats1, 
           label=target,
           cat_features=cat_cols,
           column_description=None,
           delimiter='\t',
           has_header=None,
           weight=None, 
           baseline=None,
           feature_names=None,
           thread_count=1)


## run cv to get best iteration
ctb_cv = cv(param, my_dt, fold_count=5, partition_random_seed=2017)

# fetch best round
best_round = ctb_cv['b\'Accuracy\'_test_avg'].index(np.max(ctb_cv['b\'Accuracy\'_test_avg']))

## define the classifer model
model = CatBoostClassifier(iterations=best_round, learning_rate=0.03,rsm = 0.3 ,depth=6, eval_metric='Accuracy', random_seed=2017)
	

	
## train model
model.fit(my_dt)
## make predictions
preds = model.predict(hp.test_feats1)
## make submission
sub5 = pd.DataFrame({'User_ID':hp.test.User_ID, 'Is_Response':preds})
sub5['Is_Response'] = ['happy' if x == 1 else 'not_happy' for x in sub5['Is_Response']]
sub5 = sub5[['User_ID','Is_Response']]
sub5.to_csv('sub5_cb.csv', index=False)