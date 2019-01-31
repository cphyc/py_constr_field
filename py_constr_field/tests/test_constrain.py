import numpy as np
from py_constrain.constrain import Constrain

def test_constrain():
    c = Constrain()

    c.position = np.array([0, 0, 0])
    c.operator = lambda x: x
    c.R = 10
