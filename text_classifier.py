import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


train_data_folder = 'data/classifier_train_data/'
test_data_folder = 'data/classifier_test_data/'
all_data_folder = 'data/classifier_x_val_data/'


train_data = load_files(train_data_folder, shuffle=True, random_state=42)
test_data = load_files(test_data_folder, shuffle=True, random_state=42)
all_data = load_files(all_data_folder, shuffle=True, random_state=42)

vct = CountVectorizer(stop_words='english', ngram_range=(1, 1))
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=5, random_state=42)
text_classifier = Pipeline([
    ('vect', vct),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', clf)
])

text_classifier = text_classifier.fit(train_data.data, train_data.target)
predicted = text_classifier.predict(test_data.data)

#################################

print('Prediction results:')
print('predicted', '|', 'actual', '|', 'filename')
for filename, pr_class, act_class in zip(test_data.filenames, predicted, test_data.target):
    print(pr_class, '|\t', act_class, '|\t', filename)

#################################

# evaluating metrics
print('Average prediction accuracy:', np.mean(predicted == test_data.target))
auc = roc_auc_score(predicted, test_data.target)
print('AUC:', auc)

#################################

feature_names = np.asarray(vct.get_feature_names())
top10 = np.argsort(clf.coef_[0])[-10:]
print("Top keywords for class '%s': %s" % (train_data.target_names[1], "; ".join(feature_names[top10])))

#################################
#
# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2)],
#     'tfidf__use_idf': (True, False),
#     'clf__alpha': (1e-2, 1e-3),
# }
# gs_clf = GridSearchCV(text_classifier, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(train_data.data, train_data.target)
#
# print('GridSearch suggested params:')
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
#
#################################

vct = CountVectorizer(stop_words='english', ngram_range=(1, 1))
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=5, random_state=42)
text_classifier = Pipeline([
    ('vect', vct),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', clf)
])

scores = cross_val_score(text_classifier, all_data.data, all_data.target, cv=15)
print("Cross-validated prediction accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
