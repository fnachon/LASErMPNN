from setuptools import setup, find_packages

setup(
    name='LASErMPNN',
    version='0.1.0',
    description='Inference script and utilities for the LASErMPNN protein design model.',
    author='Benjamin Fry',
    author_email='bfry@g.harvard.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'prody',
    ],
    entry_points={
        'console_scripts': [
            'run_inference=LASErMPNN.run_inference:main',
        ],
    },
    python_requires='>=3.7',
)