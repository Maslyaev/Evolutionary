#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:27:22 2020

@author: mike_ubuntu
"""

import numpy as np
from prep.derivatives import Preprocess_derivatives

if __name__ == "__main__":
    op_file_name = 'Preprocessing/Test.npy'; filename = 'Preprocessing/ssh_field.npy' 

    field = np.load(filename)
    Preprocess_derivatives(field, op_file_name, mp_poolsize=24)