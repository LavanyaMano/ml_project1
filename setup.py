from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT = '-e .'
print(HYPEN_E_DOT)

def get_requirements(file_path:str)->List[str]:
    '''
    This function will give the list from requirements.txt
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "")for req in requirements] # readlines will read newline as \n, so we are replacing with blank.
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT) #-e. in requirement automatically maps to setup.py file, so we are removing it from this list.
    
    return requirements
        

setup(
    name  = 'ML_Project1',
    version = '0.0.1',
    author= 'Lavanya',
    author_email='lavanyamkanna@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements("requirements.txt"),
    
)