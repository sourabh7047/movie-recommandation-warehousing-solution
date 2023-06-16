import numpy as np
from scipy.optimize import fmin_cg


def normalize_ratings(ratings):
    mean_ratings = np.nanmean(ratings, axis=0)
    return ratings - mean_ratings, mean_ratings


def cost(X, *args):
    num_users, num_products, num_features, ratings, mask, regularization_amount = args

    # Unroll P and Q
    P = X[0:(num_users * num_features)].reshape(num_users, num_features)
    Q = X[(num_users * num_features):].reshape(num_products, num_features)
    Q = Q.T

    # Calculate current cost
    return (np.sum(np.square(mask * (np.dot(P, Q) - ratings))) / 2) + ((regularization_amount / 2.0) * np.sum(np.square(Q.T))) + ((regularization_amount / 2.0) * np.sum(np.square(P)))


def gradient(X, *args):
    num_users, num_products, num_features, ratings, mask, regularization_amount = args

    P = X[0:(num_users * num_features)].reshape(num_users, num_features)
    Q = X[(num_users * num_features):].reshape(num_products, num_features)
    Q = Q.T

    P_grad = np.dot((mask * (np.dot(P, Q) - ratings)), Q.T) + (regularization_amount * P)
    Q_grad = np.dot((mask * (np.dot(P, Q) - ratings)).T, P) + (regularization_amount * Q.T)

    return np.append(P_grad.ravel(), Q_grad.ravel())


def low_rank_matrix_factorization(ratings, mask=None, num_features=15, regularization_amount=0.01):
    num_users, num_products = ratings.shape

    if mask is None:
        mask = np.invert(np.isnan(ratings))

    ratings = np.nan_to_num(ratings)

    np.random.seed(0)
    P = np.random.randn(num_users, num_features)
    Q = np.random.randn(num_products, num_features)

    initial = np.append(P.ravel(), Q.ravel())

    args = (num_users, num_products, num_features, ratings, mask, regularization_amount)

    X = fmin_cg(cost, initial, fprime=gradient, args=args, maxiter=3000)

    nP = X[0:(num_users * num_features)].reshape(num_users, num_features)
    nQ = X[(num_users * num_features):].reshape(num_products, num_features)

    return nP, nQ.T


def RMSE(real, predicted):
    return np.sqrt(np.nanmean(np.square(real - predicted)))