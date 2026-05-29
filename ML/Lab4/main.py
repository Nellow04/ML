import numpy
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1, 1)) * 1.0
    C = numpy.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND_fast(vrow(XPlot), m, C)))
    plt.show()

    pdfSol = numpy.load("Solution\llGAU.npy")
    pdfGau = logpdf_GAU_ND_fast(vrow(XPlot), m, C)
    print(numpy.abs(pdfSol - pdfGau).max())

    XND = numpy.load('Solution\XND.npy')
    mu = numpy.load('Solution\muND.npy')
    C = numpy.load('Solution\CND.npy')

    pdfSol = numpy.load('Solution\llND.npy')
    pdfGau = logpdf_GAU_ND_fast(XND, mu, C)
    print(numpy.abs(pdfSol - pdfGau).max())

    m_ML, C_ML = compute_mu_C(XND)
    print(m_ML)
    print(C_ML)
    print(compute_ll(XND, m_ML, C_ML))

    X1D = numpy.load('Solution\X1D.npy')
    m_ML, C_ML = compute_mu_C(X1D)
    print(m_ML)
    print(C_ML)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND_fast(vrow(XPlot), m_ML, C_ML)))
    plt.show()

    print(compute_ll(X1D, m_ML, C_ML))

    print(compute_ll(X1D, numpy.array([[1.0]]), numpy.array([[2.0]])))
    print(compute_ll(X1D, numpy.array([[0.0]]), numpy.array([[1.0]])))
    print(compute_ll(X1D, numpy.array([[2.0]]), numpy.array([[6.0]])))

