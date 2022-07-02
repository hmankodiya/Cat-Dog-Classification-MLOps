from setuptools import  setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='src',
   version='0.0.1',
   description='Small package',
   license="MIT",
   long_description=long_description,
   author='Harsh',
   author_email='hmankodiya@gmail.com',
   url="https://github.com/hmankodiya/Cat-Dog-Classification-MLOps.git",
   packages=['src'],  #same as name
   install_requires=['dvc',
                     'numpy',
                     'argparse'], 
)