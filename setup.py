from setuptools import setup

setup(
    # Common Setup
    name="isar",
    version="1.0.1",
    packages=['isar.models'],

    # Requirements
    install_requires=['numpy', 'mpctools', 'scikit-learn', 'numba', 'pandas'],

    # Meta-Data
    author='Michael P. J. Camilleri',
    author_email='michael.p.camilleri@ed.ac.uk',
    description='Implementation of the ISAR architecture as presented in '
                '"A Model for Learning Across Related Label Spaces"',
    license='GNU GPL',
    keywords='annotations learning multi-schema multi-annotator',
    url='https://github.com/michael-camilleri/ISAR-Inter_Schema_AdapteR'
)
