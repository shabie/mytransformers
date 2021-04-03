#!/usr/bin/env python
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='mytransformers',
      version='0.0.1',
      description='Makes it easier to download HuggingFace pre-trained models and upload to Kaggle if needed.',
      author='Shabie Iqbal',
      author_email='shabieiqbal@gmail.com',
      package_dir={"": "src"},
      packages=find_packages("src"),
      install_requires=required,
      license='MIT',
    )