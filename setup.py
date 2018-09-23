from pathlib import Path

from setuptools import setup, find_packages

version = Path(__file__).parents[0].joinpath(".version").read_text().split("==")[1]
long_description = Path(__file__).parents[0].joinpath("README.md").read_text()

setup(name="MultiArmedBandit-DeepLearning",
      version=version,
      author="Idan Morad",
      description="Multi Armed Bandit Algorithm W\ Deep Learning "
                  "Build using python 3.6.6 with tox 3.4.0 and tensorflow 1.10.0",
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Framework :: tox",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Education",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python :: 3.6",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Topic :: Software Development :: Testing :: Unit",
                   "Topic :: Software Development :: Version Control :: Git"],
      keywords="multi-armed-bandit deep-learning",
      packages=find_packages(exclude=['contrib', 'docs', 'tests']))
