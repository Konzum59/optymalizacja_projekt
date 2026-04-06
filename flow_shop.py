from RandomNumberGenerator import RandomNumberGenerator
import numpy as np
import math

def count_time(processing_times_array):
    for row in range(number_of_items):
        for col in range(number_of_machines):
            if row==0 and col==0:
                continue
            left = processing_times_array[row-1][col] if row else 0
            top = processing_times_array[row][col-1] if col else 0
            processing_times_array[row][col] += max(top,left)
    print(processing_times_array)
    Cmax=processing_times_array.max()
    return Cmax

if __name__ == "__main__":
    rnd = RandomNumberGenerator(831764)

    number_of_items=7
    number_of_machines=5
    items = np.array([[rnd.nextInt(1, 35) for i in range(number_of_machines)] for j in range(number_of_items)])
    best= math.inf
    best = min( best , count_time(np.copy(items)))
    print(items)
    print(best)
