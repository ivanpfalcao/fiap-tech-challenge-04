import os
from setuptools import setup, find_packages

def get_requirements(filename='requirements.txt'):
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    with open(requirements_path) as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name='stock_predictions',
    version='0.1.0',
    packages=find_packages(),
    py_modules=['predictors.stock_predictions', 'predictors.ai'],
    install_requires=get_requirements()
)