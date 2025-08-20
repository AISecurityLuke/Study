#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'pageCount' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER p
##################### 


def pageCount(n, p):
    assert 1 <= n <= 10**5, "n is out of range"
    assert 1 <= p <= n, "p is out of range"

    # Turns from the front
    front_flips = p // 2  
    # Turns from the back
    back_flips = (n // 2) - (p // 2)

    return min(front_flips, back_flips)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    p = int(input().strip())

    result = pageCount(n, p)

    fptr.write(str(result) + '\n')

    fptr.close()
