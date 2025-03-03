# purpose of this script is to give our project a name, version, and author details.
# it also automatically detects all python packages (directories containing __init__.py)
from setuptools import find_packages, setup

setup(
    name='genai_info_retrieval',
    version='0.0.2',
    author='Joackim',
    author_email='joackimagno@gmail.com',
    packages=find_packages(),
    install_requires = []
)
