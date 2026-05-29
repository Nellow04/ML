import numpy


def loaf_file (file_name):
    file = open(file_name)
    N = int(file.readline())

    light_coordinates = []

    for line in file:
        x, y = line.split()
        light_coordinates.append((int(x), int(y)))

    file.close()
    return N, light_coordinates

if __name__ == "__main__":

    N, light_coordinates = loaf_file("Data\ex5_data.txt")

    matrix = numpy.zeros((N, N))

    for x, y in light_coordinates:
        matrix[max(x - 2, 0): x + 3, max(y - 2, 0): y + 3] += 0.2
        matrix[max(x - 1, 0): x + 2, max(y - 1, 0): y + 2] += 0.3
        matrix[x, y] += 0.5
    print(matrix)