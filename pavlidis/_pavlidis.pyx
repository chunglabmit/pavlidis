'''_pavlidis.pyx - Cython implementation of Pavlidis algorithm


'''
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stdint cimport uint32_t, uint8_t

import numpy as np
cimport numpy as np

def pavlidis(uint8_t [:, :] array, int seed_row, int seed_column):
    '''Find the boundary around an object by walking in a clockwise direction

    Only works on 4-connected segments.
    Start walking from the seed pixel.

    :param array: a boolean 2d array of foreground pixels
    :param seed_row: the row (1st) coordinate of the seed pixel
    :param seed_col: the column (2nd) coordinate of the seed pixel
    '''
    cdef:
        vector[pair[size_t, size_t]] coords
        int i
        uint32_t [:, :] presult
    pavlidis_impl(array, seed_row, seed_column, coords)
    result = np.zeros((coords.size(), 2), np.uint32)
    presult = result
    for 0 <= i < coords.size():
        presult[i, 0] = coords[i].first
        presult[i, 1] = coords[i].second
    return result

cdef uint8_t get_pixel(uint8_t [:, :] array, int row, int column) nogil:
    if row < 0 or column < 0:
        return False
    if row >= array.shape[0] or column >= array.shape[1]:
        return False
    return array[row, column]

cdef void pavlidis_impl(uint8_t [:, :] array, int seed_row, int seed_column,
                        vector[pair[size_t, size_t]] &coords) nogil:
     cdef:
         int n_turns
         pair[size_t, size_t] current
         int row, column
         int direction = 0 # directions are 0 = -x, 1 = -y, 2 = x, 3 = y

     current.first = seed_row
     current.second = seed_column
     coords.push_back(current)
     n_turns = 0
     while True:
         if direction == 0: # -x
             if get_pixel(array, current.first+1, current.second-1):
                 direction = 3
                 current.first += 1
                 current.second -= 1
             elif get_pixel(array, current.first, current.second-1):
                 current.second -= 1
             elif get_pixel(array, current.first-1, current.second-1):
                 current.first -= 1
                 current.second -= 1
             else:
                 direction = 1
                 n_turns += 1
                 if n_turns == 4:
                     break
                 continue
         elif direction == 1: # -y
             if get_pixel(array, current.first-1, current.second-1):
                 direction = 0
                 current.first -= 1
                 current.second -= 1
             elif get_pixel(array, current.first-1, current.second):
                 current.first -= 1
             elif get_pixel(array, current.first-1, current.second+1):
                 current.first -= 1
                 current.second += 1
             else:
                 direction = 2
                 n_turns += 1
                 if n_turns == 4:
                     break
                 continue
         elif direction == 2: # +x
             if get_pixel(array, current.first-1, current.second+1):
                 direction = 1
                 current.first -= 1
                 current.second += 1
             elif get_pixel(array, current.first, current.second+1):
                 current.second += 1
             elif get_pixel(array, current.first+1, current.second+1):
                 current.first += 1
                 current.second += 1
             else:
                 direction = 3
                 n_turns += 1
                 if n_turns == 4:
                     break
                 continue
         else: # +y
             if get_pixel(array, current.first+1, current.second+1):
                 direction = 2
                 current.first += 1
                 current.second += 1
             elif get_pixel(array, current.first+1, current.second):
                 current.first += 1
             elif get_pixel(array, current.first+1, current.second-1):
                 current.first += 1
                 current.second -= 1
             else:
                 direction = 0
                 n_turns += 1
                 if n_turns == 4:
                     break
                 continue
         n_turns = 0
         if current.first == seed_row and current.second == seed_column:
             break
         coords.push_back(current)
