import numpy
import matplotlib.pyplot as plt
import sklearn.datasets

def vcol(x):
    return x.reshape(x.size, 1)

def vrow(x):
    return x.reshape(1, x.size)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu ) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND_fast(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5 * x.shape[0] * numpy.log(numpy.pi * 2) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * ((x - mu) * (P @ (x - mu))).sum(0)

def compute_ll(X, mu, C):
    return logpdf_GAU_ND_fast(X, mu, C).sum()

def load():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def plot_gaussian_git(D, L):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    hFea = {
        0: "Sepal length",
        1: "Sepal width",
        2: "Petal length",
        3: "Petal width"
    }

    hClasses = {
        0: ("Setosa", D0),
        1: ("Versicolor", D1),
        2: ("Virginica", D2)
    }

    for dIdx in range(4):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.ylabel("Density")

        for clsIdx in [0, 1, 2]:
            clsName, DCls = hClasses[clsIdx]

            X_feature = DCls[dIdx:dIdx + 1, :]

            mu_ML, C_ML = compute_mu_C(X_feature)

            plt.hist(X_feature.ravel(), bins=10, density=True, alpha=0.4, label=f'{clsName} Hist')

            XPlot = numpy.linspace(X_feature.min() - 1, X_feature.max() + 1, 1000)
            YPlot = numpy.exp(logpdf_GAU_ND_fast(vrow(XPlot), mu_ML, C_ML))
            plt.plot(XPlot.ravel(), YPlot, linewidth=2, label=f'{clsName} Gauss')

        plt.legend()
        plt.tight_layout()
        plt.savefig('gaussian_fit_%d.pdf' % dIdx)
    plt.show()

if __name__ == "__main__":

    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load()

    plot_gaussian_git(D, L)

# Domande

# 1)
# Sepal width ha un ottimo fit. Per tutte e tre le classi, i blocchi dell'istogramma formano una bella struttura a campana.
# Le curve Gaussiane abbracciano i dati in mdo naturale, senza forzatura. Anche Sepal length e Petal length presentano un fit accettabile.
# Le curce catturano bene il centro di massa e la dispersione generale, anche se con qualche leggera asimmetria

# 2)
# Petal width e' un cattivo fit. In particolare per la classe Setosa, dove abbiamo un istogramma formato da due pilastri altissimi con uno spaizo in mezzo.
# La curca Gaussiana viene forzata a passare in mezzo creando una collina dove in realt' la densita' reale ha un crollo.
# Anche i dati di Versicolor e Virginica sembrano molto discretizzati e poco lisci