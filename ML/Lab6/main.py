import numpy
import scipy.special
import matplotlib
import matplotlib.pyplot

from Solution.sol import uniform_cost_matrix


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_posteriors(log_class_conditional_ll, prior_array):
    logJoint = log_class_conditional_ll + vcol(numpy.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return numpy.exp(logPost)

def compute_optimal_Bayes(posterior, costMatrix):
    expectedCost = costMatrix @ posterior
    return numpy.argmin(expectedCost, 0)

def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)

    for i in range (classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1

    return M

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log((prior*Cfn) / ((1-prior) * Cfp))
    return numpy.int32(llr > th)

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize = True):
    M = compute_confusion_matrix(predictedLabels, classLabels)
    Pfn = M[0, 1] / (M[0, 1] + M[1, 1])
    Pfp = M[1, 0] / (M[0, 0] + M[1, 0])
    bayesErrpr = prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp

    if normalize:
        return bayesErrpr / numpy.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesErrpr

def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize = True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize)


def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter]  # We sort the llrs
    classLabelsSorted = classLabels[llrSorter]  # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []

    nTrue = (classLabelsSorted == 1).sum()
    nFalse = (classLabelsSorted == 0).sum()
    nFalseNegative = 0  # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse

    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)

    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    # The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    # Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    # Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
            idx]:  # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])

    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(
        thresholdsOut)  # we return also the corresponding thresholds


def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnTreshold = False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1-prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (1 - prior) * Cfp)
    idx = numpy.argmin(minDCF)

    if returnTreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

if __name__ == "__main__":

    print("Multiclass - Uniform priors and costs - Confusion Matrix")
    evalset_ll = numpy.load("Data/evalset_ll.npy")
    evalset_labels = numpy.load("Data/evalset_labels.npy")

    evalset_posteriors = compute_posteriors(evalset_ll, numpy.ones(3)/3.0)
    evalset_predictions = compute_optimal_Bayes(evalset_posteriors, uniform_cost_matrix(3))

    print (compute_confusion_matrix(evalset_predictions, evalset_labels))

    print ("Binary Task")
    evalset_llr_binary = numpy.load("Data/evalset_llr_binary.npy")
    evalset_labels_binary = numpy.load("Data/evalset_labels_binary.npy")

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        evalset_predictions_binary = compute_optimal_Bayes_binary_llr(evalset_llr_binary, prior, Cfn, Cfp)
        print(compute_confusion_matrix(evalset_predictions_binary, evalset_labels_binary))
        print()

        DCFu = compute_empirical_Bayes_risk_binary(evalset_predictions_binary, evalset_labels_binary, prior, Cfn, Cfp, normalize = False)
        print ("DCF non normalizzato: %.3f" % DCFu)

        norm_DCF = compute_empirical_Bayes_risk_binary(evalset_predictions_binary, evalset_labels_binary, prior, Cfn, Cfp, normalize = True)
        print("DCF normalizzato: %.3f" % norm_DCF)

        print()
        min_DCF, min_th = compute_minDCF_binary_fast(evalset_llr_binary, evalset_labels_binary, prior, Cfn, Cfp, returnTreshold = True)
        print("Minimum DCF: %.3f" % min_DCF)
        print("Soglia ottimale empirica: %.3f" % min_th)