# setup.py
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads dependencies from a requirements.txt file.
    Removes '-e .' if present.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name='capstone_project',
    version='0.0.1',
    author='Abhi',
    author_email='abhinavsingh121299@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)# setup.py
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads dependencies from a requirements.txt file.
    Removes '-e .' if present.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name='Space_Debris_Decay_date_prediction',
    version='0.0.1',
    author='Abhi',
    author_email='abhinavsingh121299@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)