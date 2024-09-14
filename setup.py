from setuptools import setup, find_packages

setup(
    name='ia_algorithms',
    version='0.1',
    description='Pacote para algoritmos de IA',
    author='Murilo Gonçalves Dionísio',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
    ],
    test_suite='tests',
)
