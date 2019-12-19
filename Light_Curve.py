import numpy as np 
from lightcurve_module import *

parser_args = parse_arguments()
file_name = parser_args.file
period = parser_args.period

w = 2*np.pi/period

t, obs = np.loadtxt(file_name, unpack = True, usecols = (0, 1))

c = LS_Fourier3(t, obs, period)

t_max = get_t_max(period, t[0], c)

t_av = np.arange(t_max, t_max + period, 1.0)
appr_av = get_Fourier3(t_av, period, c)

phase_obs = get_phase(t_max, period, t)
phase_av = get_phase(t_max, period, t_av)

broaden_obs_curve = extend_phase(phase_obs, obs)
broaden_av_curve = extend_phase(phase_av, appr_av)

show_curve(broaden_obs_curve, broaden_av_curve)




