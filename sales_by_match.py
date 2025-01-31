#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'sockMerchant' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY ar
#

def sockMerchant(n, ar):
    sock_counts = {}  # Dictionary to store frequency of each sock
    pairs = 0  # Variable to store the total number of pairs
    
    for sock in ar:
        if sock in sock_counts:
            sock_counts[sock] += 1
        else:
            sock_counts[sock] = 1
    
    for count in sock_counts.values():
        pairs += count // 2  # Integer division to count pairs
    
    return pairs
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    ar = list(map(int, input().rstrip().split()))

    result = sockMerchant(n, ar)

    fptr.write(str(result) + '\n')

    fptr.close()
