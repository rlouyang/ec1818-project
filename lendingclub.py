
# coding: utf-8
import calendar
import cPickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
import xgboost
get_ipython().magic(u'matplotlib inline')


# # reading in data
df = pd.read_csv('LoanStats3a.csv', header=1)
df.head()

len(df)

df.columns


# # cleaning data
# drop rows and columns with a lot of missing data
df = df.loc[df.index, df.count(0) > df.shape[0] * 0.9] # drop columns with more than 10% empty data
df = df.dropna(axis=0, how='all')
df.head()

df.shape


# ## dropping columns that can't or won't be used
nunique = df.apply(pd.Series.nunique) 
df = df.drop(nunique[nunique == 1].index, axis=1) # drop columns with only one unique value
df = df.drop(['total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', # these are recorded after the loan, so drop
              'total_rec_int', 'total_rec_late_fee', 'recoveries', 
              'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 
              'last_credit_pull_d', 'acc_now_delinq', 'delinq_amnt', 'debt_settlement_flag'], axis=1)

df = df.drop(['emp_title', 'title', 'zip_code', 'funded_amnt_inv', 'loan_amnt'], 
             axis=1) # drop employer, reason for loan, user's loan title, zip code

df = df.loc[df.count(1) > df.shape[1] * 0.7, df.columns] # drop rows with more than 30% empty data

df.head()




# ## cleaning remaining columns
df.term = df.term.str[:-7].astype(int) # drop "months" from term
df.int_rate = df.int_rate.str[:-1].astype(float) # drop percent sign from term
df.revol_util = df.revol_util.str[:-1].astype(float) # drop percent sign from term
df.loan_status = df.loan_status.str.replace('Does not meet the credit policy. Status:', 
                                            '') # drop unnecessary info from loan status
df.head()

for var in ['revol_util', 'total_acc', 'pub_rec_bankruptcies', 'tax_liens']:
    df[var] = df[var].replace(np.nan, 0)

df.shape

df.columns


# ### convert some variables to numerical
# #### ['grade', 'sub_grade', 'emp_length', 'issue_d', 'earliest_cr_line']
# convert grade and subgrade to numerical values
df.grade = df.grade.apply(ord) - 64
df.sub_grade = df.grade + df.sub_grade.str[1].apply(int) / 10.

df['has_employment'] = df.emp_length.notnull().astype(int)
df.emp_length = df.emp_length.replace('n/a', '0')
df.emp_length = df.emp_length.replace('\+? ?years?', '', regex=True)
df.emp_length = df.emp_length.replace('< 1', '0')
df.emp_length = pd.to_numeric(df.emp_length)

months = {v: k for k,v in enumerate(calendar.month_abbr)}
def get_month_number(abbr):
    return months[abbr]

df['issue_month'] = df.issue_d.str[:3].apply(get_month_number) # convert issue date to numerical values
df['issue_year'] = df.issue_d.str[-2:].astype(int)
df = df.drop(['issue_d'], axis=1)

df['earliest_cr_line_month'] = df.earliest_cr_line.str[:3].apply(get_month_number) # convert issue date to numerical values
df['earliest_cr_line_year'] = df.earliest_cr_line.str[-4:].astype(int)
df = df.drop(['earliest_cr_line'], axis=1)




# ## add dummies
for var in ['home_ownership', 'purpose', 'verification_status']:#, 'addr_state']: # dummies
    dummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([var], axis=1)

df = df.drop(['addr_state'], axis=1)

df.head()

df.to_csv('LoanStats3a_cleaned.csv')


# # read in clean data
df = pd.read_csv('LoanStats3a_cleaned.csv', index_col=0)
df.head()


# # splitting into X, Y, train, test
# output
Y = df['loan_status'] == 'Fully Paid'
# input
X = df.drop(['loan_status'], axis=1)

# Split data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)


# # classification evaluation metrics
def calculate_roc_curve(Y_true, Y_prob, label='ROC curve'):    
    fpr, tpr, _ = roc_curve(Y_true, Y_prob)
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

def calculate_confusion_matrix(Y_true, Y_pred):    
    cm=confusion_matrix(Y_true, Y_pred)
    print(cm)
    
def log_likelihood(Y_true, Y_prob):
    return np.mean((1 - Y_true) * np.log(1 - Y_prob) + Y_true * np.log(Y_prob))

def brier_score(Y_true, Y_prob):
    return np.mean(np.square(Y_true - Y_prob))

def calculate_calibration_curve(Y_true, Y_prob, n_bins=10, label='calibration curve'):
    y, x = calibration_curve(Y_true, Y_prob, n_bins=n_bins)
    plt.plot(x, y, 's-', label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('predicted probability')
    plt.ylabel('fraction of positives')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')

def calculate_prob_hist(Y_prob, label='probability histogram'):
    plt.hist(Y_prob, bins=10, range=(0, 1), density=True, histtype="step", label=label, lw=2)
    plt.xlabel('predicted probability')
    plt.ylabel('frequency')
    plt.title('Probability Histogram')
    plt.legend(loc='upper left')


# # Classification
# Logistic Regression Classifier
logistic = LogisticRegression()
logistic.fit(X_train, Y_train)
logistic_preds = logistic.predict(X_val)
logistic_score = logistic.score(X_val, Y_val)
logistic_score

logistic_probs = logistic.predict_proba(X_val)[:,1]
calculate_roc_curve(Y_val, logistic_probs, 2)
print roc_auc_score(Y_val, logistic_probs)

confusion_matrix(Y_val, logistic_preds)

# workaround
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# statsmodels for feature extraction
import statsmodels.api as sm
logit = sm.Logit(Y_train, X_train)
model = logit.fit()
model.summary()

with open('logistic_output.tex', 'w') as f:
    f.write(model.summary().as_latex())

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf.fit(X_train, Y_train)
rf_preds = rf.predict(X_val)
rf_score = rf.score(X_val, Y_val)
rf_score

rf_probs = rf.predict_proba(X_val)[:,1]
print roc_auc_score(Y_val, rf_probs)

confusion_matrix(Y_val, rf_preds)

# ExtraTrees Classifier
et = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1)
et.fit(X_train, Y_train)
et_preds = et.predict(X_val)
et_score = et.score(X_val, Y_val)
et_score

et_probs = et.predict_proba(X_val)[:,1]
print roc_auc_score(Y_val, et_probs)

confusion_matrix(Y_val, et_preds)

# XGBoost
xgb = xgboost.XGBClassifier(n_estimators=100, n_jobs=-1, learning_rate=0.2)
xgb.fit(X_train, Y_train)
xgb_preds = xgb.predict(X_val)
xgb_score = xgb.score(X_val, Y_val)
xgb_score

xgb_probs = xgb.predict_proba(X_val)[:,1]
print roc_auc_score(Y_val, xgb_probs)

confusion_matrix(Y_val, xgb_preds)

# MLP (relu for now)
mlp = MLPClassifier(hidden_layer_sizes=(20, 10), activation='relu')
mlp.fit(X_train, Y_train)
mlp_preds = mlp.predict(X_val)
mlp_score = mlp.score(X_val, Y_val)
mlp_score

mlp_probs = mlp.predict_proba(X_val)[:,1]
print roc_auc_score(Y_val, mlp_probs)

confusion_matrix(Y_val, mlp_preds)

voters = [
#           ('et', ExtraTreesClassifier(n_estimators=1000, n_jobs=-1)),
          ('rf', RandomForestClassifier(n_estimators=1000, n_jobs=-1)),
          ('xgb', xgboost.XGBClassifier(n_estimators=100, n_jobs=-1)),
#           ('rf-b', RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')),
#           ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), activation='relu')),
         ]

voting_weights = np.array([ 
#                            et_score, 
                           rf_score, 
                           xgb_score,
#                            w_rf_score, mlp_score
                          ])

voting = VotingClassifier(estimators=voters, voting='soft', weights=list(voting_weights ** 1), n_jobs=-1)
voting.fit(X_train, Y_train)
voting_preds = voting.predict(X_val)
voting.score(X_val, Y_val)

voting_probs = voting.predict_proba(X_val)[:,1]
print roc_auc_score(Y_val, voting_probs)

confusion_matrix(Y_val, voting_preds)

predictions = {'logistic': logistic_probs, 
               'random forest': rf_probs, 
               'extremely randomized trees': et_probs, 
               'XGBoost': xgb_probs,
               'neural network': mlp_probs, 
               'voting': voting_probs
              }
models = ['neural network', 'logistic', 'extremely randomized trees', 'random forest', 'XGBoost', 'voting']
for key in models:
    calculate_calibration_curve(Y_val, predictions[key], label=key)
plt.savefig('calibration_curve.pdf')
plt.show()

for key in models:
    calculate_prob_hist(predictions[key], label=key)
plt.savefig('prob_hist.pdf')
plt.show()

for key in models:
    calculate_roc_curve(Y_val, predictions[key], label=key)
plt.savefig('roc_curve.pdf')
plt.show()

for key in models:
    print(key, brier_score(Y_val, predictions[key]))
for key in models:
    print(key, log_likelihood(Y_val, predictions[key]))

# data to plot
n_groups = len(models)
brier_score_list = [brier_score(Y_val, predictions[key]) for key in models]
 
# create plot
index = np.arange(n_groups)
 
plt.bar(index, brier_score_list, 
        label='Brier score')
 
models[2] = 'extremely\nrandomized\ntrees'
plt.xlabel('Model')
plt.ylabel('Brier score')
plt.ylim(ymin=0.1)
plt.title('Brier Scores for Classification Models')
plt.xticks(index, models)
 
plt.savefig('classification_brier.pdf')
plt.show()

models[models.index('extremely\nrandomized\ntrees')] = 'extremely randomized trees'
auc_list = [roc_auc_score(Y_val, predictions[key]) for key in models]
 
# create plot
index = np.arange(len(models))
 
plt.bar(index, auc_list, 
        label='AUC')
 
models[2] = 'extremely\nrandomized\ntrees'
plt.xlabel('Method')
plt.ylabel('AUC Score')
plt.ylim(ymin=0.50)
plt.title('AUC Scores for Classification Models')
plt.xticks(index, models)

plt.savefig('classification_auc.pdf')
plt.show()

models.remove('neural network')
models[models.index('extremely\nrandomized\ntrees')] = 'extremely randomized trees'
log_likelihood_list = [-log_likelihood(Y_val, predictions[key]) for key in models]
 
# create plot
index = np.arange(len(models))
 
plt.bar(index, log_likelihood_list, 
         label='Negative Log-Likelihood')
 
models[1] = 'extremely\nrandomized\ntrees'
plt.xlabel('Method')
plt.ylabel('Negative Log-Likelihood')
plt.ylim(ymin=0.35)
plt.title('Negative Log-Likelihood for Classification Models')
plt.xticks(index, models)

plt.savefig('classification_loglik.pdf')
plt.show()

rf_scores = []
for n_trees in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1)
    rf.fit(X_train, Y_train)
    rf_probs = rf.predict_proba(X_val)[:,1]
    rf_scores.append(roc_auc_score(Y_val, rf_probs))

plt.semilogx([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
             rf_scores)
plt.xlabel('Number of Trees in Random Forest (Log Scale)')
plt.ylabel('AUC Score (Accuracy)')
plt.title('Random Forest Classifier: Number of Trees and Accuracy')
plt.savefig('num_trees_rfc.pdf')

xgb_scores = []
for n_trees in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
    xgb = xgboost.XGBClassifier(n_estimators=n_trees, n_jobs=-1)
    xgb.fit(X_train, Y_train)
    xgb_probs = xgb.predict_proba(X_val)[:,1]
    xgb_scores.append(roc_auc_score(Y_val, xgb_probs))

plt.semilogx([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
             xgb_scores)
plt.xlabel('Number of Trees in XGBoost (Log Scale)')
plt.ylabel('AUC Score (Accuracy)')
plt.title('XGBoost Classifier: Number of Trees and Accuracy')
plt.savefig('num_trees_xgbc.pdf')




# # feature importances
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for i in range(X.shape[1]):
    print("%d & %s & %f \\\\" % (i + 1, X.columns[indices[i]], importances[indices[i]]))

df.columns


# # predicting the interest rate (regression)

# ## regression evaluation metrics
def mse(actual, predicted):
    return np.mean(np.square(actual - predicted))

def mae(actual, predicted):
    return np.mean(np.absolute(actual - predicted))

Y = df['int_rate']
X = df.drop(['int_rate', 'loan_status', 'grade', 'sub_grade', 'installment'], axis=1)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)

# OLS from statsmodels
linear = sm.OLS(Y_train, X_train)
model = linear.fit()
model.summary()

with open('linear_output.tex', 'w') as f:
    f.write(model.summary().as_latex())

# Linear Regression 
linear = LinearRegression()
linear.fit(X_train, Y_train)
linear_preds = linear.predict(X_val)
linear_score = linear.score(X_val, Y_val)
print linear_score

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rf.fit(X_train, Y_train)
rf_preds = rf.predict(X_val)
rf_score = rf.score(X_val, Y_val)
print rf_score

# ExtraTrees Regressor
et = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1)
et.fit(X_train, Y_train)
et_preds = et.predict(X_val)
et_score = et.score(X_val, Y_val)
print et_score

# MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu')
mlp.fit(X_train, Y_train)
mlp_preds = mlp.predict(X_val)
mlp_score = mlp.score(X_val, Y_val)
print mlp_score

# XGBoost
xgb = xgboost.XGBRegressor(n_estimators=10000, n_jobs=-1)
xgb.fit(X_train, Y_train)
xgb_preds = xgb.predict(X_val)
xgb_score = xgb.score(X_val, Y_val)
print xgb_score

mse(Y_val, xgb_preds), mae(Y_val, xgb_preds)

voting_preds = (rf_preds * rf_score + xgb_preds * xgb_score) / (rf_score + xgb_score)
voting_score = np.corrcoef(Y_val, voting_preds)[0][1] ** 2
voting_score

predictions = {'linear': linear_preds, 
               'random forest': rf_preds, 
               'extremely randomized trees': et_preds, 
               'XGBoost': xgb_preds,
               'voting': voting_preds}
models = ['linear', 'extremely randomized trees', 'random forest', 'XGBoost', 'voting']
for key in models:
    print(key, mse(Y_val, predictions[key]))
for key in models:
    print(key, mae(Y_val, predictions[key]))

n_groups = len(models)
mse_list = [mse(Y_val, predictions[key]) for key in models]
mae_list = [mae(Y_val, predictions[key]) for key in models]
 
# create plot
index = np.arange(n_groups)
bar_width = 0.35
 
plt.bar(index, mse_list, bar_width,
        label='MSE')
plt.bar(index + bar_width, mae_list, bar_width,
        label='MAE')
 
models[1] = 'extremely\nrandomized\ntrees'
plt.xlabel('Model')
plt.ylabel('Error')
plt.title('Accuracy Measures for Regression Models')
plt.xticks(index + bar_width / 2., models)
plt.legend()
 
plt.savefig('regression_errors.pdf')
plt.show()

models[models.index('extremely\nrandomized\ntrees')] = 'extremely randomized trees'
r2_list = [np.corrcoef(Y_val, predictions[key])[0][1] ** 2 for key in models]
 
# create plot
index = np.arange(n_groups)
 
plt.bar(index, r2_list, 
        label='R2')
 
models[1] = 'extremely\nrandomized\ntrees'
plt.xlabel('Model')
plt.ylabel('R2')
plt.title('R2 for Regression Models')
plt.xticks(index, models)

plt.savefig('regression_r2.pdf')
plt.show()


# ## feature importances (regression)
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for i in range(X.shape[1]):
    print("%d & %s & %f \\\\" % (i + 1, X.columns[indices[i]], importances[indices[i]]))


# ## hyperparameter tuning (regression)
rf_scores = []
for n_trees in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
    rf.fit(X_train, Y_train)
    rf_preds = rf.predict(X_val)
    rf_scores.append(mse(Y_val, rf_preds))

plt.semilogx([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
             rf_scores)
plt.xlabel('Number of Trees in Random Forest (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.title('Random Forest Regressor: Number of Trees and Accuracy')
plt.savefig('num_trees_rfr.pdf')

xgb_scores = []
for n_trees in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
    xgb = xgboost.XGBRegressor(n_estimators=n_trees, n_jobs=-1)
    xgb.fit(X_train, Y_train)
    xgb_preds = xgb.predict(X_val)
    xgb_scores.append(mse(Y_val, xgb_preds))

plt.semilogx([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
             xgb_scores)
plt.xlabel('Number of Trees in XGBoost (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.title('XGBoost Regressor: Number of Trees and Accuracy')
plt.savefig('num_trees_xgbr.pdf')
