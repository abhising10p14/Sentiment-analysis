# let's check cross validation score of the model
# cv score acts a unbiased estimate of models accuracy on unseen data

# NaiveBayes
import happy_2 as hp

mod1 = GaussianNB()
target = hp.train_feats['Is_Response']
## Naive Bayes 1
print(cross_val_score(mod1, hp.train_feats1, target, cv=5, scoring=make_scorer(accuracy_score)))

## Naive Bayes 2 - tfidf is giving higher CV score
print(cross_val_score(mod1,hp.train_feats2, target, cv=5, scoring=make_scorer(accuracy_score)))

# make our first set of predictions

clf1 = GaussianNB()
clf1.fit(hp.train_feats1, target)

clf2 = GaussianNB()
clf2.fit(hp.train_feats2, target)

preds1 = clf1.predict(test_feats1)
preds2 = clf2.predict(test_feats2)

def to_labels(x):
    if x == 1:
        return "happy"
    return "not_happy"

sub1 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds1})
sub1['Is_Response'] = sub1['Is_Response'].map(lambda x: to_labels(x))

sub2 = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':preds2})
sub2['Is_Response'] = sub2['Is_Response'].map(lambda x: to_labels(x))



sub1 = sub1[['User_ID', 'Is_Response']]
sub2 = sub2[['User_ID', 'Is_Response']]


## write submission files
sub1.to_csv('sub1_cv.csv', index=False)
sub2.to_csv('=sub2_tf.csv', index=False)

