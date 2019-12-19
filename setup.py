from setuptools import setup

setup(
	name='LightCurve',

    version='0.0.1',

    author='Anna Fedoteva',

    author_email='fedotieva2010@yandex.ru',

    description='Аппроксимация кривой блеска',

    py_modules = ['lightcurve_module'],
    scripts=['Light_Curve.py'],
    test_suite = 'lightcurve_module_testing.py',
    install_requires=['numpy>=1.13', 'matplotlib>=2.0'],
    keywords='astronomy',

    )