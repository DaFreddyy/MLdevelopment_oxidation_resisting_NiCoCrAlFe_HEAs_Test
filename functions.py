import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# GradientBoostingRegressor
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,cross_validate
from sklearn.metrics import make_scorer

def data_split(trainset,element_names,per):
############trainset and testset split using holdouttest
    i = 0
    testset = pd.DataFrame()#columns=trainset.columns
    count = int(trainset.shape[0]*per/2/len(element_names))+1
    while i < count:
      for element in element_names:
        max_element = trainset[element].max()
        max_index = trainset[element].idxmax()

        min_element = trainset[element].min()
        min_index = trainset[element].idxmin()

        min_row=trainset.loc[min_index:min_index,:]
        max_row=trainset.loc[max_index:max_index,:]
        if max_index !=min_index:
          testset = pd.concat([testset, min_row], ignore_index=True)
          testset = pd.concat([testset, max_row], ignore_index=True)
          trainset = trainset.drop(min_index)
          trainset = trainset.drop(max_index)
        if max_index ==min_index:
          testset = pd.concat([testset, min_row], ignore_index=True)
          trainset = trainset.drop(min_index)
      i =i+1
    return trainset,testset

def data_sampling(trainset,comp_major_low,comp_major_high,comp_major_inter,
                 comp_minor_low,comp_minor_high,comp_minor_inter,
                 T_low,T_high,T_inter,size,element_names,random_state=None):
    ##################################################################################
    comp_major_bins = np.arange(comp_major_low, comp_major_high, comp_major_inter)
    comp_minor_bins = np.arange(comp_minor_low, comp_minor_high, comp_minor_inter)
    temp_bins = np.arange(T_low, T_high, T_inter)
    temp_bins  = 1000/(temp_bins+273)
    temp_bins.sort()
    ##################################################################################
    major_feature_names = ['Ni','Co','Cr','Al','Fe'] ###major element name
    minor_feature_names = set(element_names) - set(major_feature_names)
    minor_feature_names = list(minor_feature_names) ###minor element name
    ##################################################################################
    for compy in major_feature_names:
        trainset[compy+'_Bin'] = pd.cut(trainset[compy],bins=comp_major_bins, labels=range(len(comp_major_bins)-1))
    for compy in minor_feature_names:
        trainset[compy+'_Bin'] = pd.cut(trainset[compy],bins=comp_minor_bins, labels=range(len(comp_minor_bins)-1))
    trainset['Test_Temperature_Bin'] = pd.cut(trainset['invT'],bins=temp_bins, labels=range(len(temp_bins)-1))
    all_bin_cols = [col for col in trainset.columns if '_Bin' in col]
    size = size
    replace = True  # with replacement
    rng = np.random.default_rng(random_state)
    fn = lambda obj: obj.loc[rng.choice(obj.index, size, replace),:]
    Sampled_trainset = trainset.groupby(all_bin_cols, as_index=False, group_keys=False).apply(fn)
    ##################################################################################################
    keep_cols = [col for col in Sampled_trainset.columns if '_Bin' not in col]
    Sampled_trainset = Sampled_trainset[keep_cols]
    return Sampled_trainset

# Function to calculate Euclidean distance
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2)**2))

# Apply the function to each group
def calculate_distance(group):
    if len(group) == 1:
        group['Distance'] = None
        # return 0
    elif len(group) == 2:
        row1 = group.iloc[0, 1:][['Ni', 'Cr', 'Co', 'Al', 'Fe']].values
        row2 = group.iloc[1, 1:][['Ni', 'Cr', 'Co', 'Al', 'Fe']].values
        group['Distance'] = euclidean_distance(row1, row2 )
    else:
        # return np.nan
        group['Distance'] = None
    return group

# Apply the function to each group
def finalscore(group):
    alpha = 0.5
    beta = 1-alpha
    if len(group) == 1:
        print('wrong single phase alloy')
    elif len(group) == 2:
        s1 = group.iloc[0, 1:]['Phase_Score']#.values
        s2 = group.iloc[1, 1:]['Phase_Score']#.values
        s3 = group.iloc[0, 1:]['Distance_Rank']#.values

        group['Final Score'] = (s1+s2)*beta+s3*alpha
    else:
         print('wrong multiple phase alloy')
    return group

def build_feature_row(composition, feature_names, temperature_C=None):
    """Build a single-row feature DataFrame for model.predict() from a raw steel
    composition (element fractions in at%) and the test temperature.

    composition: dict, e.g. {'Ni': 50, 'Cr': 20, 'Co': 12, 'Al': 12, 'Fe': 6}.
        Elements not listed default to 0 (absent from the alloy).
    feature_names: model.feature_names_in_.tolist() -- the exact columns/order the
        trained model expects.
    temperature_C: oxidation test temperature in degrees Celsius. Required if the
        model uses the 'invT' feature.
    """
    row = pd.Series(0.0, index=feature_names, dtype=float)

    for element, pct in composition.items():
        if element in row.index:
            row[element] = pct

    if 'invT' in row.index:
        if temperature_C is None:
            raise ValueError("This model uses 'invT' -- pass temperature_C (in degrees C).")
        row['invT'] = 1000 / (temperature_C + 273)

    # composition-derived engineered features (only set if the model actually uses them)
    if 'Al+Cr' in row.index:
        row['Al+Cr'] = composition.get('Al', 0) + composition.get('Cr', 0)
    if 'Cr/Al' in row.index:
        al = composition.get('Al', 0)
        row['Cr/Al'] = composition.get('Cr', 0) / al if al else 0

    return pd.DataFrame([row])

def predict_composition(composition, model, temperature_C=None):
    """Predict the oxidation rate constant from a steel composition and test temperature.

    composition: dict of element -> at% (elements not listed are treated as 0 at%).
    model: a GradientBoostingRegressor trained on composition (+ invT) features.
    temperature_C: test temperature in degrees C (required if the model uses 'invT').
    Returns (log10_kp, kp).
    """
    feature_names = model.feature_names_in_.tolist()

    usable = set(feature_names)
    if {'Al+Cr', 'Cr/Al'} & usable:
        usable |= {'Al', 'Cr'}  # these also feed the engineered columns
    ignored = [e for e in composition if e not in usable]
    if ignored:
        print(f"Warning: these elements are not model features and are ignored: {ignored}")

    predictdf = build_feature_row(composition, feature_names, temperature_C)
    log_kp = model.predict(predictdf)[0]
    return log_kp, 10 ** log_kp

def leakfree_cv(make_model, trainset, element_names, composition_features,
                sampling_params, n_splits=5, seed=42):
    """Leak-free K-fold CV for the oxidation-kp pipeline.

    The naive approach -- resample the whole trainset with data_sampling (bootstrap
    WITH replacement) and then KFold on it -- leaks: identical duplicated rows land
    in both the train and validation folds, inflating the CV score.

    Here, for each fold we split the ORIGINAL (un-resampled) trainset, apply
    data_sampling ONLY to the training rows of that fold, fit make_model() on the
    resampled train fold, and score on the un-resampled validation rows. This mirrors
    the final held-out test (train on resampled data, evaluate on original data) and
    contains no duplication leakage.

    make_model: zero-arg callable returning a fresh, unfitted estimator.
    trainset:   original (un-resampled) training DataFrame; must contain 'kp',
                the composition_features and the element columns data_sampling bins on.
    sampling_params: dict of the comp/T binning args for data_sampling
                (everything except trainset, element_names, random_state).
    Returns {'test_MAE','test_MSE','test_R2'} arrays (one entry per fold), matching
    the keys produced by sklearn's cross_validate with those scorers.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    positions = np.arange(len(trainset))
    maes, mses, r2s = [], [], []
    for tr_pos, va_pos in kf.split(positions):
        tr_fold = trainset.iloc[tr_pos].copy()   # copy: data_sampling adds _Bin columns in place
        va_fold = trainset.iloc[va_pos]
        sampled_tr = data_sampling(tr_fold, **sampling_params,
                                   element_names=element_names, random_state=seed)
        cols = [c for c in composition_features if c in sampled_tr.columns]
        X_tr = sampled_tr[cols]
        y_tr = np.log10(sampled_tr['kp'])
        X_va = va_fold[cols]
        y_va = np.log10(va_fold['kp'])
        model = make_model()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, pred))
        mses.append(mean_squared_error(y_va, pred))
        r2s.append(r2_score(y_va, pred))
    return {'test_MAE': np.array(maes),
            'test_MSE': np.array(mses),
            'test_R2': np.array(r2s)}