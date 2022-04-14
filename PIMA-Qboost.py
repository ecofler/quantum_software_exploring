import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import model_selection

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dimod
from dwave.system import LeapHybridSampler

from tabulate import tabulate

import itertools

from imblearn.over_sampling import SMOTE


class DecisionStumpClassifier:
    """Decision tree classifier that operates on a single feature with a single splitting rule.

    The index of the feature used in the decision rule is stored
    relative to the original data frame.
    """
    
    def __init__(self, X, y, feature_index):
        """Initialize and fit the classifier.

        Args:
            X (array): 
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself uses only a single feature.
            y (array):
                1D array of class labels, as ints.  Labels should be
                +/- 1.
            feature_index (int):
                Index or pair of indexes for the feature(s) used by the weak classifier,
                relative to the overall data frame.
        """
        

        self.i = feature_index
        if type(self.i) == type(1):
            self.clf = DecisionTreeClassifier(max_depth=1)
            self.clf.fit(X[:, [self.i]], y)
        else:
            self.clf = DecisionTreeClassifier(max_depth=2)
            self.clf.fit(np.concatenate(([X[:, self.i[0]]], [X[:, self.i[1]]]), axis=0).T , y)
            

    def predict(self, X):
        """Predict class.

        Args:
            X (array):
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself will make a prediction based only a single
                feature.

        Returns:
            Array of class labels.
        """
        if type(self.i) == type(1):
            return self.clf.predict(X[:, [self.i]])
        else:
            return self.clf.predict(np.concatenate(([X[:, self.i[0]]], [X[:, self.i[1]]]), axis=0).T)   

def _build_H(classifiers, X, output_scale):
    """Construct matrix of weak classifier predictions on given set of input vectors."""
    H = np.array([clf.predict(X) for clf in classifiers], dtype=float).T

    # Rescale H
    H *= output_scale

    return H


class EnsembleClassifier:
    """Ensemble of weak classifiers."""

    def __init__(self, weak_classifiers, weights, weak_classifier_scaling, offset=1e-9):
        """Initialize ensemble from list of weak classifiers and weights.

        Args:
            weak_classifiers (list):
                List of classifier instances.
            weights (array):
                Weights associated with the weak classifiers.
            weak_classifier_scaling (float):
                Scaling for weak classifier outputs.
            offset (float):
                Offset value for ensemble classifier.  The default
                value is a small positive number used to prevent
                ambiguous 0 predictions when weak classifiers exactly
                balance each other out.
        """
        self.classifiers = weak_classifiers
        self.w = weights
        self.weak_clf_scale = weak_classifier_scaling
        self.offset = offset

    def predict(self, X):
        """Compute ensemble prediction.

        Note that this function returns the numerical value of the
        ensemble predictor, not the class label.  The predicted class
        is sign(predict()).
        """
        H = _build_H(self.classifiers, X, self.weak_clf_scale)

        # If we've already filtered out those with w=0 and we are only
        # using binary weights, this is just a sum
        preds = np.dot(H, self.w)
        return preds - self.offset

    def predict_class(self, X):
        """Compute ensemble prediction of class label."""
        preds = self.predict(X)

        # Add a small perturbation to any predictions that are exactly
        # 0, because these will not count towards either class when
        # passed through the sign function.  Such zero predictions can
        # happen when the weak classifiers exactly balance each other
        # out.
        preds[preds == 0] = 1e-9

        return np.sign(preds)

    def score(self, X, y):
        """Compute accuracy score on given data."""
        if sum(self.w) == 0:
            # Avoid difficulties that occur with handling this below
            return 0.0
        return accuracy_score(y, self.predict_class(X))

    def squared_error(self, X, y):
        """Compute squared error between predicted and true labels.

        Provided for testing purposes.
        """
        p = self.predict(X)
        return sum((p - y)**2)

    def fit_offset(self, X):
        """Fit offset value based on class-balanced feature vectors.

        Currently, this assumes that the feature vectors in X
        correspond to an even split between both classes.
        """
        self.offset = 0.0
        # Todo: review whether it would be appropriate to subtract
        # mean(y) here to account for unbalanced classes.
        self.offset = np.mean(self.predict(X))

    def get_selected_features(self):
        """Return list of features corresponding to the selected weak classifiers."""
        return [clf.i for clf, w in zip(self.classifiers, self.w) if w > 0]




def _build_bqm(H, y, lam):
    """Build BQM.

    Args:
        H (array):
            2D array of weak classifier predictions.  Each row is a
            sample point, each column is a classifier.
        y (array):
            Outputs
        lam (float):
            Coefficient that controls strength of regularization term
            (larger values encourage decreased model complexity).
    """
    n_samples = np.size(H, 0)
    n_classifiers = np.size(H, 1)

    # samples_factor is a factor that appears in front of the squared
    # loss term in the objective.  In theory, it does not affect the
    # problem solution, but it does affect the relative weighting of
    # the loss and regularization terms, which is otherwise absorbed
    # into the lambda parameter.

    # Using an average seems to be more intuitive, otherwise, lambda
    # is sample-size dependent.
    samples_factor = 1.0 / n_samples

    bqm = dimod.BQM('BINARY')
    bqm.offset = samples_factor * n_samples

    for i in range(n_classifiers):
        # Note: the last term with h_i^2 is part of the first term in
        # Eq. (12) of Neven et al. (2008), where i=j.
        bqm.add_variable(i, lam - 2.0 * samples_factor *
                         np.dot(H[:, i], y) + samples_factor * np.dot(H[:, i], H[:, i]))

    for i in range(n_classifiers):
        for j in range(i+1, n_classifiers):
            # Relative to Eq. (12) from Neven et al. (2008), the
            # factor of 2 appears here because each term appears twice
            # in a sum over all i,j.
            bqm.add_interaction(
                i, j, 2.0 * samples_factor * np.dot(H[:, i], H[:, j]))

    return bqm


def _minimize_squared_loss_binary(H, y, lam):
    """Minimize squared loss using binary weight variables."""
    bqm = _build_bqm(H, y, lam)

    sampler = LeapHybridSampler()
    results = sampler.sample(bqm, label='Example - QBoost')
    weights = np.array(list(results.first.sample.values()))
    energy = results.first.energy

    return weights, energy


    """Initialize and fit QBoost classifier.

        X should already include all candidate features (e.g., interactions).

        Args:
            X (array):
                2D array of feature vectors.
            y (array):
                1D array of class labels (+/- 1).
            lam (float):
                regularization parameter.
            weak_clf_scale (float or None):
                scale factor to apply to weak classifier outputs.  If
                None, scale by 1/num_classifiers.
            drop_unused (bool):
                if True, only retain the nonzero weighted classifiers.
        """

class QBoostClassifier(EnsembleClassifier):
    """Construct an ensemble classifier using quadratic loss minimization.    """

    def __init__(self, X, y, lam, weak_clf_scale=None, drop_unused=True, dictionary=1):
      
        if not all(np.isin(y, [-1, 1])):
            raise ValueError("Class labels should be +/- 1")

        num_features = np.size(X, 1)

        if weak_clf_scale is None:
            weak_clf_scale = 1 / num_features
        if dictionary == 1:
            wclf_candidates = [DecisionStumpClassifier(X, y, i) for i in range(num_features)]
        else:
            wclf_candidates = [DecisionStumpClassifier(X, y, p) for p in itertools.combinations(range(num_features) , 2)] + [DecisionStumpClassifier(X, y, i) for i in range(num_features)]
             

        H = _build_H(wclf_candidates, X, weak_clf_scale)

        # For reference, store individual weak classifier scores.
        self.weak_scores = np.array([np.mean(np.sign(h) * y > 0) for h in H.T])

        weights, self.energy = _minimize_squared_loss_binary(H, y, lam)

        # Store only the selected classifiers
        if drop_unused:
            weak_classifiers = [wclf for wclf, w in zip(
                wclf_candidates, weights) if w > 0]
            weights = weights[weights > 0]
        else:
            weak_classifiers = wclf_candidates

        super().__init__(weak_classifiers, weights, weak_clf_scale)
        self.fit_offset(X)

        # Save candidates so we can provide a baseline accuracy report.
        self._wclf_candidates = wclf_candidates

    def report_baseline(self, X, y):
        """Report accuracy of weak classifiers.

        This provides context for interpreting the performance of the boosted
        classifier.
        """
        scores = np.array([accuracy_score(y, clf.predict(X))
                           for clf in self._wclf_candidates])
        data = [[len(scores), scores.min(), scores.mean(), scores.max(), scores.std()]]
        headers = ['count', 'min', 'mean', 'max', 'std']

        print('Accuracy of weak classifiers (score on test set):')
        print(tabulate(data, headers=headers, floatfmt='.3f'))


def qboost_lambda_sweep(X, y, lambda_vals, val_fraction=0.4, verbose=False, **kwargs):
    """Run QBoost using a series of lambda values and check accuracy against a validation set.

    Args:
        X (array):
            2D array of feature vectors.
        y (array):
            1D array of class labels (+/- 1).
        lambda_vals (array):
            Array of values for regularization parameter, lambda.
        val_fraction (float):
            Fraction of given data to set aside for validation.
        verbose (bool):
            Print out diagnostic information to screen.
        kwargs:
            Passed to QBoost.__init__.

    Returns:
        QBoostClassifier:
            QBoost instance with best validation score.
        lambda:
            Lambda value corresponding to the best validation score.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction)

    best_score = -1
    best_lambda = None
    best_clf = None

    if verbose:
        print('{:7} {} {}:'.format('lambda', 'n_features', 'score'))

    for lam in lambda_vals:
        qb = QBoostClassifier(X_train, y_train, lam, **kwargs)
        score = qb.score(X_val, y_val)
        if verbose:
            print('{:<7.4f} {:<10} {:<6.3f}'.format(
                lam, len(qb.get_selected_features()), score))
        if score > best_score:
            best_score = score
            best_clf = qb
            best_lambda = lam

    return best_clf, lam



DATASET_PATH = 'datasets/'

data_path = os.path.join(DATASET_PATH, 'PIMA_diabetes_presence_dataset.csv')
dataset = pd.read_csv(data_path, skiprows=1, header=None)

dataset.columns = [
    "NumTimesPrg", "PlGlcConc", "BloodP",
    "SkinThick", "TwoHourSerIns", "BMI",
    "DiPedFunc", "Age", "HasDiabetes"]

print(dataset.shape)


Index_Mis_Val = ['BMI','BloodP','PlGlcConc','SkinThick','TwoHourSerIns']
for ind in Index_Mis_Val:
    # Calculate the median value for BMI
    median_bmi = dataset[ind].median()
    # Substitute it in the BMI column of the
    # dataset where values are 0
    dataset[ind] = dataset[ind].replace(to_replace=0, value=median_bmi)


#Oversampling with SMOTE (data is unbalanced) -- it should be combined with some undersampling (see https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

from collections import Counter

sm = SMOTE(random_state=42)
dataset_X = dataset
dataset_X = dataset_X.drop("HasDiabetes", axis=1)
dataset_Y = dataset["HasDiabetes"].copy()
dataset_X_new, dataset_Y_new = sm.fit_resample(dataset_X, dataset_Y)
dataset_new = pd.concat([dataset_X_new,dataset_Y_new], axis=1)
dataset = dataset_new

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(
    dataset, test_size=0.2, random_state=42)

train_set_labels = train_set["HasDiabetes"].copy()
train_set = train_set.drop("HasDiabetes", axis=1)

test_set_labels = test_set["HasDiabetes"].copy()
test_set = test_set.drop("HasDiabetes", axis=1)


# Prepare the configuration to run the test

seed = 7
results = []
names = []
X = train_set.to_numpy()
Y = train_set_labels.to_numpy()

X_test = test_set.to_numpy()
Y_test = test_set_labels.to_numpy()

# We choose the Random Forest Classifier and AdaBoost to compare with QBoost

model = RandomForestClassifier(n_estimators=8, max_depth=1)

kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
cv_result = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

msg = "%s: %f (%f)" % ("Random Forest Classifier", cv_result.mean(), cv_result.std())

print(msg)



classical_ensamble = AdaBoostClassifier(n_estimators=8, random_state=0)

cv_result = model_selection.cross_val_score(classical_ensamble, X, Y, cv=kfold, scoring='accuracy')

msg = "%s: %f (%f)" % ("AdaBoost Ensamble Classifier", cv_result.mean(), cv_result.std())

print(msg)

# We apply Qboost to DS with depth 1 and 2 (by default, both AdaBoost and Qboost are applied on Decision Stumps, so Decision Trees
# with depth = 1) with the lambda sweep 
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
for i in range(len(Y)):
    if Y[i] == 0:
        Y[i]-=1
for i in range(len(Y_test)):
    if Y_test[i] == 0:
        Y_test[i] -= 1
#print(len(test_set_labels),len(Y))
#for i in range(len(test_set_labels)):

#print(Y.iloc[0:20])

#n_features = len(X[0,:])
#normalized_lambdas = np.linspace(0.0, 0.5, 10)
#lambdas = normalized_lambdas / n_features
#print('Performing cross-validation using {} values of lambda, this may take several minutes...'.format(len(lambdas)))
#quantum_ensamble, lam = qboost_lambda_sweep(X, Y, lambdas)

lam = 0.0625
quantum_ensamble = QBoostClassifier(X, Y, lam)

quantum_ensamble.score(X_test, Y_test)
number_feat_or_trees = quantum_ensamble.get_selected_features()
print('Selected lambda:', lam)
print('Selected features:', number_feat_or_trees)
print('Selected trees: ', [plot_tree(quantum_ensamble.classifiers[i].clf) for i in range(len(quantum_ensamble.classifiers)) ] ) 
print('Score on test set: {:.3f}'.format(quantum_ensamble.score(X_test, Y_test) ))

