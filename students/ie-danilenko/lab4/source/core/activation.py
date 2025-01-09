import numpy as np

class Null:
    def get(input):
        return input
    
    def dget(input):
        return 1.0
    
class Sign:
    def get(x):
        return np.where(x > 0, 1., np.where(x < 0, -1., 0))
