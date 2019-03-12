# ISAR (Inter-Schema AdapteR)

This repository contains python code for replicating the Results in the Paper:
> "A Model for Learning Across Related Label Spaces", Under Review

**Due to the nature of our collaboration with MRC Harwell, we can only provide the data (and code) for the simulation components of the study.**

## Contents

1. Repository Structure
1. Installation and Setting Up
1. Reproducing the Experiments

## Repository Structure

At the top level, there are two `requirements` files, for easy installation of the necessary packages. All the code is packaged under the python directory which contains the following packages:
 * **Tools**: Contains in-house libraries/modules used by the remainder of the scripts: this includes a multi-processing wrapper for exploiting multi-core architectures and extensions to some numpy methods.
 * **Models**: Contains the ISAR implementations for the Multinomial Class-Conditional and Inter-Annotator Variability models.
 * **20 Newsgroups**: Contains scripts for replicating the results of the simulations on the 20 Newsgroups Dataset (i.e. Section 5.2 in the Paper)
 
In general, the replication scripts follow the pattern: `Load`, `Learn`, `Visualise`.

## Installation and Setting Up

The repository is designed to be a self-contained implementation, subject to some library requirements

### Requirements

The code is designed to be run with Python 3 and will not work with Python 2. It is reccomended to set up a python virtual environment (example using [Conda](https://conda.io/en/latest/)). The code was tested and runs with Python 3.6.

The code also makes use of the following libraries: for simplicity, we provide conda and pip requirements file for automatically installing the specified version of the library to guarantee compatibility:
 * scikit-learn
 * matplotlib
 * pathos
 * pandas
 * numpy

### Installation Procedure

1. Download or clone this repository
1. From the command line, change to the location of the repository contents, or open a terminal directory in the location.
1. Install conda-based requirements:
  ```bash
  conda install --file conda.req
  ```
1. Install pip-based requirements (the `pathos` library is not available under conda)
  ```bash
  pip install -r pip.req
  ```

That is all!



