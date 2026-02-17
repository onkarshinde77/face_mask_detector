from setuptools import setup,find_packages
from typing import List

hypen_e = "-e ."
def get_dependency(file:str)->List :
    lib = []
    with open(file,'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                lib.append(str(line))
        if hypen_e in lib:
            lib.remove(hypen_e)
    return lib

setup(
    name = "Entry-Exist-Monitoring-of-Vehicle",
    version= '1.0.0',
    author='onkar shinde',
    author_email='onkarshinde.ai@gmail.com',
    packages=find_packages(),
    install_requires = get_dependency('requirements.txt')
)