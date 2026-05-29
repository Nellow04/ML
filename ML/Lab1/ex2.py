import sys


class Bus:
    def __init__(self, busId, lineId, x, y, time):
        self.busId = busId
        self.lineId = lineId
        self.x = x
        self.y = y
        self.time = time

def load_all_records(file_name):
    try:
        records_list = []
        with open(file_name) as file:
            for line in file:
                busId, lineId, x, y, time = line.split()
                record = Bus(busId, lineId, int(x), int(y), int(time))
                records_list.append(record)

        return  records_list
    except:
        raise

def euclidean_distance(r1, r2):
    return ((r1.x - r2.x)**2 + (r1.y - r2.y)**2)**0.5

def compute_bus_distance_time(records_list, busId):
    bus_records = sorted([i for i in records_list if i.busId == busId], key= lambda x: x.time)

    if len(bus_records) == 0:
        return None, None

    total_distance = 0.0
    for previous_record, current_record in zip(bus_records[:-1], bus_records[1:]):
        total_distance += euclidean_distance(current_record, previous_record)

    total_time = bus_records[-1].time - bus_records[0].time
    return total_distance, total_time

def compute_line_avg_speed(records_list, lineId):
    records_list_filtered = [i for i in records_list if i.lineId == lineId]
    bus_set = set([i.busId for i in records_list_filtered])

    if len(bus_set) == 0:
        return 0.0

    total_distance = 0.0
    total_time = 0.0

    for busId in bus_set:
        distance, time = compute_bus_distance_time(records_list_filtered, busId)
        total_time += time
        total_distance += distance

    return total_distance/total_time

if __name__ == "__main__":

    records_list = load_all_records(sys.argv[1])

    if sys.argv[2] == "-b":
        print("%s Total distance: " % sys.argv[3], compute_bus_distance_time(records_list, sys.argv[3])[0])
    elif sys.argv[2] == "-l":
        print("%s Average speed" % sys.argv[3], compute_line_avg_speed(records_list, sys.argv[3]))
    else:
        raise KeyError()