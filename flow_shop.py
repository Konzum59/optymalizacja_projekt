from RandomNumberGenerator import RandomNumberGenerator
import numpy as np
import math

def count_time(array, permutation):
    processing_times_array=array[permutation]
    for row in range(1, number_of_items):
        processing_times_array[row][0]+=processing_times_array[row-1][0]
    for col in range(1, number_of_machines):
            processing_times_array[0][col]+=processing_times_array[0][col-1]
    for row in range(1 , number_of_items):
        for col in range(1 ,number_of_machines):

            left = processing_times_array[row-1][col]
            top = processing_times_array[row][col-1]
            processing_times_array[row][col] += max(top,left)
    print(processing_times_array)
    Cmax=processing_times_array.max()
    return Cmax
if __name__ == "__main__":
    rnd = RandomNumberGenerator(831764)

    number_of_items=4
    number_of_machines=3
    items = np.array([[rnd.nextInt(1, 35) for i in range(number_of_machines)] for j in range(number_of_items)])
    new_order = [2, 0, 1, 3]
    best = math.inf ,new_order
    best = [min( best[0] , count_time(items, new_order)), new_order]
    print(items)
    print(best[0], best[1])
