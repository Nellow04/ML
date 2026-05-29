import numpy


class Competitor:
    def __init__(self, name, surname, country):
        self.name = name
        self.surname = surname
        self.country = country

def load_competitors(file_name):

    file = open(file_name)
    n = int(file.readline())

    scores_list = []
    competitors_list = []

    for index, line in enumerate(file):
        name, surname, country = line.split()[0:3]
        scores = line.split()[3:]
        scores = [float(i) for i in scores]
        scores_list.append(scores)
        competitors_list.append(Competitor(name, surname, country))

    score_matrix = numpy.array((scores_list))

    file.close()
    return competitors_list, score_matrix

if __name__ == "__main__":

    competitors_list, score_matrix = load_competitors("Data/ex8_data.txt")

    tot_score_array = score_matrix.sum(1)
    max_score_array = score_matrix.max(1)
    min_score_array = score_matrix.min(1)

    comp_scores_array = tot_score_array - max_score_array - min_score_array

    alternative_solution = numpy.array(score_matrix)
    alternative_solution.sort(axis=1)
    alternative_solution_comp_scores_array = alternative_solution[:, 1:-1]
    alternative_solution_comp_scores_array = alternative_solution_comp_scores_array.sum(1)

    print("Debug - comp_scores_array vs alternative_comp_scores_array")
    print(comp_scores_array)
    print(alternative_solution_comp_scores_array)
    print()

    index = numpy.argsort(comp_scores_array)
    best_list = index[-3:][::-1]

    print ("Final ranking:")
    for i in best_list:
        print("%d: %s %s - Score: %.1f" % (i+1, competitors_list[i].name, competitors_list[i].surname, comp_scores_array[i]))
    print()