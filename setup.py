from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.txt'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='training_dashboard',
    version='0.1.0',
    description='No bullshit, dead simple training visualizer for tf-keras.',
    url='https://github.com/vibhuagrawal14/training_dashboard',
    author='Vibhu Agrawal',
    author_email='vibhu.agrawal14@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=['training_dashboard'],
    install_requires=['numpy', 'pandas', 'bqplot', 'tensorflow'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
