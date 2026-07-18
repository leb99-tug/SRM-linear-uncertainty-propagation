import skrf as rf
import numpy as np
#import scipy as sc
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

from skrf.media import Coaxial,MLine
from scipy.signal import savgol_filter
import time
import metas_unclib as unc

from aux_functions import load_cov, cov_to_std, sample_complex_normal
num_simulations = 5000

start_time = time.time()


cov, f, name = load_cov("unc_network")
std_nw = cov_to_std(cov) 
cov, f, name = load_cov("unc_open_p1")
std_o1 = cov_to_std(cov) 
cov, f, name = load_cov("unc_open_p2")
std_o2 = cov_to_std(cov) 
cov, f, name = load_cov("unc_short_p1")
std_s1 = cov_to_std(cov) 
cov, f, name = load_cov("unc_short_p2")
std_s2 = cov_to_std(cov) 
cov, f, name = load_cov("unc_match_p1")
std_m1 = cov_to_std(cov) 
cov, f, name = load_cov("unc_match_p2")
std_m2 = cov_to_std(cov) 

cov, f, name = load_cov("unc_open_def")
std_odef = cov_to_std(cov) 
cov, f, name = load_cov("unc_short_def")
std_sdef = cov_to_std(cov) 
cov, f, name = load_cov("unc_match_def")
std_mdef = cov_to_std(cov) 


print(std_nw)
 
short_p1 = rf.Network('MeasurementData/Short_2Port_raw.s2p')['0-40GHz'].s11
short_p2 = rf.Network('MeasurementData/Short_2Port_raw.s2p')['0-40GHz'].s22
open_p1 = rf.Network('MeasurementData/Open_2Port_raw.s2p')['0-40GHz'].s11
open_p2 = rf.Network('MeasurementData/Open_2Port_raw.s2p')['0-40GHz'].s22
load_p1 = rf.Network('MeasurementData/Match_2Port_raw.s2p')['0-40GHz'].s11
load_p2 = rf.Network('MeasurementData/Match_2Port_raw.s2p')['0-40GHz'].s22
reciprocal = rf.Network('MeasurementData/Line_6_raw.s2p')['0-40GHz']
thru = rf.Network('MeasurementData/Line_0_raw.s2p')['0-40GHz']

FWST = rf.Network('MeasurementData/FWST_Line_0.s1p')['0-40GHz']
RWST = rf.Network('MeasurementData/RWST_Line_0.s1p')['0-40GHz']

step_imp = rf.Network('MeasurementData/Impedance_raw.s2p')['0-40GHz']

match_def = rf.Network('match_0402.s1p')['0-40GHz']
short_def = rf.Network('short_pcb.s1p')['0-40GHz']
open_def = rf.Network('open_pcb.s1p')['0-40GHz']

reciprocal_def = rf.Network('MeasurementData/Line_6.s2p')['0-40GHz']

freq = short_p1.frequency

for i in range(num_simulations):


    Short = np.zeros((len(freq),2,2),dtype=complex)
    Short[0:len(freq),0,0] = sample_complex_normal(short_p1.s[:,0,0], std_s1)
    Short[0:len(freq),1,1] = sample_complex_normal(short_p2.s[:,0,0], std_s2)
    Short = rf.Network(frequency=freq,s=Short)

    Match = np.zeros((len(freq),2,2),dtype=complex)
    Match[0:len(freq),0,0] = sample_complex_normal(load_p1.s[:,0,0], std_m1)
    Match[0:len(freq),1,1] = sample_complex_normal(load_p2.s[:,0,0], std_m2)
    Match = rf.Network(frequency=freq,s=Match)

    Open = np.zeros((len(freq),2,2),dtype=complex)
    Open[0:len(freq),0,0] = sample_complex_normal(open_p1.s[:,0,0], std_o1)
    Open[0:len(freq),1,1] = sample_complex_normal(open_p2.s[:,0,0], std_o2)
    Open = rf.Network(frequency=freq,s=Open)

    Reciprocal_nw = np.zeros((len(freq),2,2),dtype=complex)
    Reciprocal_nw[0:len(freq),0,0] = sample_complex_normal(reciprocal.s[:,0,0], std_nw[:,0,0])
    Reciprocal_nw[0:len(freq),0,1] = sample_complex_normal(reciprocal.s[:,0,1], std_nw[:,0,1])
    Reciprocal_nw[0:len(freq),1,0] = sample_complex_normal(reciprocal.s[:,1,0], std_nw[:,1,0])
    Reciprocal_nw[0:len(freq),1,1] = sample_complex_normal(reciprocal.s[:,1,1], std_nw[:,1,1])
    Reciprocal_nw = rf.Network(frequency=freq,s=Reciprocal_nw)
    
    Short_def = np.zeros((len(freq),2,2),dtype=complex)
    Short_def[0:len(freq),0,0] = sample_complex_normal(short_def.s[:,0,0], std_sdef)
    Short_def[0:len(freq),1,1] = sample_complex_normal(short_def.s[:,0,0], std_sdef)
    Short_def = rf.Network(frequency=freq,s=Short_def)

    Match_def = np.zeros((len(freq),2,2),dtype=complex)
    Match_def[0:len(freq),0,0] = sample_complex_normal(match_def.s[:,0,0], std_mdef)
    Match_def[0:len(freq),1,1] = sample_complex_normal(match_def.s[:,0,0], std_m2)
    Match_def = rf.Network(frequency=freq,s=Match_def)

    Open_def = np.zeros((len(freq),2,2),dtype=complex)
    Open_def[0:len(freq),0,0] = sample_complex_normal(open_def.s[:,0,0], std_odef)
    Open_def[0:len(freq),1,1] = sample_complex_normal(open_def.s[:,0,0], std_odef)
    Open_def = rf.Network(frequency=freq,s=Open_def)

    
    SOLT_AB = rf.calibration.UnknownThru(ideals=[Open_def,Short_def,Match_def,reciprocal_def],
                                        measured=[Open,Short,Match,Reciprocal_nw],
                                        switch_terms=[FWST,RWST])

    SOLT_AB.run()


    DUT = SOLT_AB.apply_cal(step_imp)

    
    DUT.write_touchstone(f"SOLR_Folder/DUT_{i}.s2p")

    if i == 0: 
        one_cycle_time = time.time() - start_time
    print(f"Simulation {i} of {num_simulations} completed.")
    print(f"Estimated time remaining: {(num_simulations-i)*one_cycle_time:.2f} seconds.")

