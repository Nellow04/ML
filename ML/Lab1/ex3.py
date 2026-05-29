import sys

highest_month_names = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

if __name__ == "__main__":

    try:
        file = open(sys.argv[1], "r")
    except:
        print("Error while opening the file")
        sys.exit(0)

    highest_cities = {}
    highest_months = {}

    for line in file:
        name, surname, city, date = line.split()
        day, month, year = date.split("/")
        month_int = int(month)

        if city not in highest_cities:
            highest_cities[city] = 0
        highest_cities[city] += 1

        if month_int not in highest_months:
            highest_months[month_int] = 0
        highest_months[month_int] += 1

    file.close()

    print("Births per city: ")
    for city in highest_cities:
        print("\t%s: %d" % (city, highest_cities[city]))

    print("Birhs per month: ")
    for month in sorted(highest_months):
        print("\t%s: %d" % (highest_month_names[month], highest_months[month]))

    print("Average number of births: %.2f" % (float(sum(highest_cities.values()))/float(len(highest_cities.keys()))))