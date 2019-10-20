#Программа аппроксимирует наблюдаемую кривую блеска рядом Фурье до 3-го порядка и строит фазовую кривую.

import numpy as np 
from matplotlib import pyplot as plt 

#Функция находит момент максимума блеска
def jd_max(period, jd_0, c):
	jd = np.arange(jd_0 - period, jd_0, 0.01)
	w = 2*np.pi/period
	func = c[0] + c[1]*np.cos(w*jd) + c[2]*np.sin(w*jd) + c[3]*np.cos(2*w*jd) + c[4]*np.sin(2*w*jd) + c[5]*np.cos(3*w*jd) + c[6]*np.sin(3*w*jd)
	return jd[np.where(func == np.min(func))] 

#Функция переводит даты наблюдений в фазы
def get_phase(jd_max, P, jd):
        num_P = (jd - jd_max)/P
        return num_P - np.trunc(num_P)

file_name = 'obs_data.dat'
period = 355.2
w = 2*np.pi/period

t, obs = np.loadtxt(file_name, unpack = True, usecols = (0, 1))

N = len(t)
n = 7              #Число слагаемых в разложении
CONST = np.ones(N)
ct = np.cos(w*t)
st = np.sin(w*t)
c2t = np.cos(2*w*t)
s2t = np.sin(2*w*t)
c3t = np.cos(3*w*t)
s3t = np.sin(3*w*t)
array = np.array([CONST, ct, st, c2t, s2t, c3t, s3t])

#Находим коэф-ты разложения с помощью МНК
big_array = np.array([])
little_array = np.array([])
for i in range(0, n):
	big_array = np.append(big_array, (np.tile(array[i], n)).reshape(n, N)*array)
	little_array = np.append(little_array, obs*array[i])

big_array = big_array.reshape(n, n, N)
little_array = little_array.reshape(n, N)

matrix = np.sum(big_array, 2)
right_part = np.sum(little_array, 1)

c = np.linalg.solve(matrix, right_part)

jd_max = jd_max(period, t[0], c)

jd_av = np.arange(jd_max, jd_max + period, 1.0)
appr_av = c[0] + c[1]*np.cos(w*jd_av) + c[2]*np.sin(w*jd_av) + c[3]*np.cos(2*w*jd_av) + c[4]*np.sin(2*w*jd_av) + c[5]*np.cos(3*w*jd_av) + c[6]*np.sin(3*w*jd_av)

phase_obs = get_phase(jd_max, period, t)
phase_av = get_phase(jd_max, period, jd_av)

phase_obs_neg = np.array([])
phase_obs_over1 = np.array([])

phase_av_neg = np.array([])
phase_av_over1 = np.array([])

obs_neg = np.array([])
obs_over1 = np.array([])

av_neg = np.array([])
av_over1 = np.array([])

for i in range(0, len(phase_obs)):
	if phase_obs[i]<=0.25 and phase_obs[i]>0:
		phase_obs_over1 = np.append(phase_obs_over1, phase_obs[i] + 1)
		obs_over1 = np.append(obs_over1, obs[i])
	if phase_obs[i]>=0.75 and phase_obs[i]<1:
		phase_obs_neg = np.append(phase_obs_neg, phase_obs[i] - 1)
		obs_neg = np.append(obs_neg, obs[i])
	
for i in range(0, len(phase_av)):
	if phase_av[i]<=0.25 and phase_av[i]>0:
		phase_av_over1 = np.append(phase_av_over1, phase_av[i] + 1)
		av_over1 = np.append(av_over1, appr_av[i])
	if phase_av[i]>=0.75 and phase_av[i]<1:
		phase_av_neg = np.append(phase_av_neg, phase_av[i] - 1)
		av_neg = np.append(av_neg, appr_av[i])

obs_ = np.concatenate([obs_neg, obs, obs_over1])
phase_obs_ = np.concatenate([phase_obs_neg, phase_obs, phase_obs_over1])

appr_av_ = np.concatenate([av_neg, appr_av, av_over1])
phase_av_ = np.concatenate([phase_av_neg, phase_av, phase_av_over1])


fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.plot(phase_av_, appr_av_, 'k-')
ax.plot(phase_obs_, obs_, 'b.')
ax.tick_params(top = 'on', right = 'on', direction = 'in', which = 'both')
ax.set_xlim(-0.3, 1.3)
ax.set_xticks(np.arange(-0.3, 1.3, 0.05), minor = 'True')
ax.set_xlabel('$\\varphi$')
ax.set_ylabel('Magnitude')

ax.invert_yaxis()


plt.show()



