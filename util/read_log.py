import os
from config.config import settings_system


def read_price(month_: int, day_: int):
    path = settings_system['logging_path']

    cost = 0

    with open(path, 'r') as f:
        for line in f:
            time_start_index = line.index('.py - ')
            time_end_index = line.index(' - INFO')
            time = line[time_start_index + 6:time_end_index]
            month = int(time.split(' ')[0][2:4])
            day = int(time.split(' ')[0][4:])
            #print(f'{month}:{day}')
            if month != month_:
                continue

            if day <= day_:
                continue

            if "INFO: cost: " not in line:
                continue

            start_index = line.index('cost: ')
            end_index = line.index('. prompt tokens:')
            cost += float(line[start_index + 6:end_index])

    return cost


if __name__ == '__main__':
    print(read_price(9, 23))
