import sys
from time import process_time_ns


def compute_avg_score(scores_list):
    return sum(sorted(scores_list) [1:-1])

class Competitor:
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores
        self.avg_score = compute_avg_score(self.scores)

if __name__ == "__main__":

    best_competitors_list = []
    highest_country_scores = {}

    with open(sys.argv[1]) as file:
        for line in file:
            name, surname, country = line.split()[0:3]
            scores = line.split()[3:]
            scores = [float(i) for i in scores]

            competitor = Competitor(
                name, surname, country, scores
            )

            best_competitors_list.append(competitor)

            if len(best_competitors_list) > 4:
                best_competitors_list = sorted(best_competitors_list, key = lambda i: i.avg_score)[::-1][0:3]

            if competitor.country not in highest_country_scores:
                highest_country_scores[competitor.country] = 0

            highest_country_scores[competitor.country] += competitor.avg_score

    if len(highest_country_scores) == 0:
        print("There are no competitors")
        sys.exit(0)

    best_country = None
    for country_check in highest_country_scores:
        if best_country is None or highest_country_scores[country_check] > highest_country_scores[best_country]:
            best_country = country_check

    print ("Final ranking:")
    for pos, competitor in enumerate(best_competitors_list):
        print("%d: %s %s - Score: %1.f" % (pos+1, competitor.name, competitor.surname, competitor.avg_score))
    print()
    print("Best Country:")
    print("%s - Total score: %.1f" % (best_country, highest_country_scores[best_country]))