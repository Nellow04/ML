def loaf_file(file_name):
    file = open(file_name)
    N = int(file.readline())
    light_coordinates = []

    for line in file:
        x, y = line.split()
        light_coordinates.append((int(x), int(y)))

    file.close()

    return N, light_coordinates

def matrix_list_zeros(N):
    matrix = []

    for i in range(N):
        matrix.append([0 for j in range(N)])

    return matrix

def matrix_dict_zeros(N):
    matrix = {}

    for i in range(N):
        for j in range(N):
            matrix[(i, j)] = 0

    return matrix

def print_matrix_list(matrix):
    for row in matrix:
        for col in row:
            print("%.1f" % col, end=" ")
        print()

def print_matrix_dict(matrix, n):
    for i in range(n):
        for j in range(n):
            print("%.1f" % matrix[(i, j)], end=" ")
        print()

if __name__ == "__main__":

    N, light_coordinates = loaf_file("Data\ex5_data.txt")
    matrix_list = matrix_list_zeros(N)
    matrix_dict = matrix_dict_zeros(N)

    for x, y in light_coordinates:
        for dx in range (-2, 3):
            for dy in range (-2, 3):
                if x + dx >= 0 and x + dx < N and y+dy >= 0 and y + dy < N:
                    matrix_list[x + dx][y + dy] += 0.2
                    matrix_dict[(x + dx, y + dy)] += 0.2

        for dx in range (-1, 2):
            for dy in range (-1, 2):
                if x + dx >= 0 and x + dx < N and y+dy >= 0 and y + dy < N:
                    matrix_list[x + dx][y + dy] += 0.3
                    matrix_dict[(x + dx, y + dy)] += 0.3

        matrix_list[x][y] += 0.5
        matrix_dict[(x, y)] += 0.5

    print_matrix_list(matrix_list)
    print()
    print_matrix_dict(matrix_dict, N)
