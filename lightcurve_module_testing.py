import unittest
import lightcurve_module as lc 
import numpy as np

class Test_lc_module(unittest.TestCase):

	def test_t_max(self):
		t_0 = 3*np.pi
		period = 2*np.pi
		c = np.array([0, 1, 0, 0, 0, 0, 0])

		self.assertEqual(lc.get_t_max(period, t_0, c), np.pi)


	def test_phase(self):
		t = np.array([15])
		t_max = 10
		P = 10
		
		self.assertEqual(lc.get_phase(t_max, P, t), np.array([0.5]))


	def test_LS_Fourier3(self):
		period = 2*np.pi
		t_0 = 3*np.pi
		t = np.arange(t_0 - period, t_0, 0.01)
		w = 2*np.pi/period
		value = np.cos(w*t)
		c_true = np.array([0, 1, 0, 0, 0, 0, 0])
		c = lc.LS_Fourier3(t, value, period)
		
		self.assertTrue(np.equal(c_true, c).all())


	def test_extend_phase(self):
		phase = np.array([0.1, 0.8])
		result = lc.extend_phase(phase, phase)
		phase_ex = np.around(result['Phase'], decimals = 3)
		value_ex = np.around(result['Value'], decimals = 3)
		phase_ex_true = np.array([-0.2, 0.1, 0.8, 1.1])
		value_ex_true = np.array([0.8, 0.1, 0.8, 0.1])
		test_result = np.concatenate([np.equal(phase_ex_true, phase_ex), np.equal(value_ex_true, value_ex)])
	
		self.assertTrue(test_result.all())


if __name__ == '__main__':
    unittest.main()