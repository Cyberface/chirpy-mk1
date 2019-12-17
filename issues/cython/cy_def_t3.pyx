#import chirpy_mk1

#from chirpy_mk1.utils import G_Newt, c_ls, MSUN_SI, MTSUN_SI, MRSUN_SI, PC_SI, GAMMA, Msun_to_sec

#import numpy as np

def cython_array_in_out(arrin):
    arrout = [a*a for a in arrin]
    return arrout


#def cython_def_array_in_out(float arrin):
#    cdef float arrout[]
#    arrout = [a*a for a in arrin]
#    return arrout

#arrin=[1,2,3,4]
#cython_array_in_out(arrin)
#cython_def_array_in_out(arrin)
