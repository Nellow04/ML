import sklearn.datasets
import numpy

def load_iris():
    return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"]

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):
    mu, C = compute_mu(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D

if __name__ == "__main__":

    D, L = load_iris()
    mu, C = compute_mu(D)

    print(mu)
    print(C)

    P = compute_pca(D, m = 4)
    print(P)

    P_sol = numpy.load("Solution/IRIS_PCA_matrix_m4.npy")
    print(P_sol)



