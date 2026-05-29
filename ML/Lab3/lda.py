import numpy
import scipy.linalg


def load_iris():

    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape ((1, x.size))

def compute_mu(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0

    mu_global = vcol(D.mean(1))

    for i in numpy.unique(L):
        D_labels = D[:, L == i]
        mu = vcol(D_labels.mean(1))
        Sb += (mu - mu_global) @ (mu - mu_global).T * D_labels.shape[1]
        Sw += (D_labels - mu) @ (D_labels - mu).T

    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L ,m):
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)

    return U[:, ::-1][:, 0:m]

def apply_lda(U, D):
    return U.T @ D

if __name__ == "__main__":
    D, L = load_iris()
    U = compute_lda_geig(D, L, m = 2)

    print(U)
    print(compute_lda_geig(D, L, m = 2))

    U_sol = numpy.load("Solution/IRIS_LDA_matrix_m2.npy")
    print(U_sol)

    print(numpy.linalg.svd(numpy.hstack([U, U_sol]))[1])

