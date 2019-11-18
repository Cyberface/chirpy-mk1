#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages


# https://stackoverflow.com/questions/26900328/install-dependencies-from-setup-py
import os
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(name='chirpy-mk1',
      version='0.0.1',
      description='Standalone time domain model for the gravitational wave signal from non-spinning binary black hole mergers',
      author='Sebastian Khan',
      author_email='KhanS22@Cardiff.ac.uk',
      packages=find_packages(),
      install_requires=install_requires,
      url='https://github.com/Cyberface/chirpy-mk1'
     )
