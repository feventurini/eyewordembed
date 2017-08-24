"""
DISCLAIMER FOR PLAGIARISM:
this script is taken from the second answer to this stack overflow post:
https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution/1557906
"""

import atexit
from time import time
from datetime import timedelta

def secondsToStr(t):
    return str(timedelta(seconds=t))

line = "="*40
def log(s, elapsed=None):
    print(line)
    print(secondsToStr(time()), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()

def endlog():
    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

def now():
    return secondsToStr(time())

start = time()
atexit.register(endlog)
log("Start Program")