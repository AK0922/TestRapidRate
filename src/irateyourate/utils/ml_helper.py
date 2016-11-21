from sklearn import linear_model, svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from src.irateyourate.utils import log_helper

log = log_helper.get_logger("ML_Helper")


def train_linear_model(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    linear_reg_model = linear_model.LinearRegression()
    linear_reg_model.fit(x_train, y_train)

    log.info("Linear Regression accuracy: " + str(linear_reg_model.score(x_test, y_test)))

    return linear_reg_model


def train_svm(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    svm_regressor = svm.LinearSVR()
    svm_regressor.fit(x_train, y_train)

    log.info("SVR accuracy: " + str(svm_regressor.score(x_test, y_test)))

    return svm_regressor


def extract_training_parameters(doc2vec_model, sentiment_scores_dict):
    x_docvecs = list()
    y_scores = list()

    for tag in sentiment_scores_dict.keys():
        x_docvecs.append(doc2vec_model.docvecs[tag])
        y_scores.append(sentiment_scores_dict[tag])

    return x_docvecs, y_scores


def persist_model_to_disk(model, model_path):
    joblib.dump(model, model_path)


def get_model_from_disk(model_path):
    return joblib.load(model_path)
