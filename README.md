# ISAR (Inter-Schema AdapteR)

This repository contains python code for replicating the Results in the Paper:
> "The Extended Dawid-Skene Model: Fusing Information from Multiple Data Schemas", Accepted for oral presentation at ECMLPKDD Workshop on Automating Data Science

as well as the Extended Dawid-Skene model defined therein to fuse information from multiple annotators labelling under different schemas. 
If you find the code useful in any of your projects, please consider citing the above paper.
 
**Due to the nature of our collaboration with MRC Harwell, we can only provide the data for the simulation components of the study: we do however provide the code to replicate all results.**

## Contents

1. Repository Structure
1. Installation and Setting Up
1. Reproducing the Experiments

## Repository Structure

At the top level, there are two `requirements` files, for easy installation of the necessary packages (except for the custom mpctools library: instructions for this are provided below). 
There are 3 high-level directories:
 * **isar**: Contains the classes implementing the two models in our paper, the baseline DS and Extended DS using our ISAR adapter, under the `models` package.
 * **scripts**: Contains the scripts for replicating the results in the paper.
 * **data**: Directory which at the outset contains the parameters for the MRC-Harwell-trained models which form the basis of our simulations. This is also the default directory for storing results from simulation runs (but can be configured, see below).
 
In general, the replication scripts are split into two stages: one for simulating/training the models, and the other for visualising results.

## Installation and Setting Up

The repository is designed to be a self-contained implementation, subject to some standard library requirements

### Requirements

The code is designed to be run with Python 3 and will not work with Python 2. It is recommended to set up a python 
virtual environment (example using [Conda](https://conda.io/en/latest/)). The code was tested and runs with Python 3.6.

The code also makes use of the following libraries: for simplicity, we provide conda and pip requirements file for 
automatically installing the specified version of the library to guarantee compatibility:
 * scikit-learn
 * matplotlib
 * *pathos*
 * pandas
 * numpy
 * numba
 
All the above, apart from *pathos* are available through conda: pathos needs to be installed through pip (all 
instructions given below). We also use our own externally packaged `mpctools` library. Instructions are provided below for setting it up.

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
1. Download the mpctools library from https://github.com/michael-camilleri/mpctools and install (from within the top-level directory):
  ```bash
  python setup.py sdist --format=tar
  pip install dist/mpctools-<version>.tar
  ```
  (replace `<version>` with the correct version).

If you just want to replicate the results in the paper and that is all that is required. If however you wish to make use of the isar modules in another
project we provide a setup script for installing it. Again, this is as simple as running the provided `setup.py` script:
  ```bash
  python setup.py sdist --format=tar
  pip install dist/isar-1.0.1.tar
  ```

## Replicating the Results

All scripts below can be run from the command line by navigating to the directory containing the appropriate script and running it through python:

```bash
python <SCRIPT_NAME.py> [COMMAND LINE ARGUMENTS]
```

All the scripts support some form of tweaking by way of command line arguments passed through as key-value pairs. Typically, the default values generate the figures/tables exactly as they appear in the paper: where this needs to be run multiple times with different configurations, the settings for each run are provided. However, feel free to experiment with the values as this can help identify interesting avenues. Calling the script with the `-h` key generates a helpful summary of what each parameter controls.

We will now describe replication of the results in each relevant section

### Likelihood-based Evaluation on the real data (Section 5.2)

Due to the ownership of the data, we cannot publicly provide the data for this section (do get in touch if you are interested in exploring it however). We do however provide the code used to generate the results, showing how we calculate metrics and the assumptions we do. And obviously, if one has access to another data-source with the same modality, the results can be compared.

We assume a dataset stored as a Pandas dataframe (in msgpack format). The index columns are:
 * The Fold Index (Letters, in our case A through K)
 * The Run Number: This splits the samples into contiguous runs (in our case, 0 through 62)
 * The Mouse Identifier (unique, in our case RFID)
 * Time-Stamp (absolute time since epoch)
As regards Data Columns:
 * Schema: set in I, II, III, V (other values can be considered if you specify the schemas/names appropriately: the script must be modified accordingly)
 * Segment: A 3-Letter alphabetical segment identifier
 * Annotator Labels: Albl.1 through Albl.12 (skipping 11). These are CDType 0 through 13 (12 behaviours + NIS). The NIS
    distinguishes between informative unlabelling and missing data according to Section 4 (pg 7)

Given the above, Table 1 can be replicated in a two-step process:

1. Train the DS/ISAR Models on the true dataset. This will generate two files, one each for DS and ISAR: just pass `none` to the -o parameter if you wish to suppress any simulation component. For a complete replication, execute the below:
   ```bash
   python Learn_MRC_DATA.py -i ../data/mrc_data.df -o ../data/Learn_DS ../data/Learn_ISAR -r 0
   ```
2. Visualise the Results, in Tabular Form:
   ```bash
   python Visualise_MRC_Data.py -r ../data/Learn_DS.npz ../data/Learn_ISAR.npz -s I II III IV
   ```
   Note that the `-s` flag is only as a way of labelling the tables.

### Latent State Inference in Synthethic Data (Section 5.3)

With the synthethic data we sought to capture the data generation process as close as possible to what we learnt above. Tables 2/4 can be replicated in a two-step process.

1. Train the DS/ISAR Models on the entire dataset. This will generate two files, one each for DS and ISAR: just pass `none` to the -o parameter if you wish to suppress any simulation component. The below is the code to generate the results for the `Realistic` case (which is also the default): note that this will take about 6 Hours since it has not **yet** been optimised to run multiple runs in parallel.
   ```bash
   python Simulate_Predictions.py -o ../data/Compare_DS ../data/Compare_ISAR -r 0 -n 0 20 -l 60 5400 -f 10 -s 13 15 17 10 -p true
   ```
   In all cases, the following parameters are common:
    * `-r 0` (Random State)
    * `-n 0 20` (Start index and number of independent runs)
    * `-f 10` (Number of folds)
   The remaining settings for each run are as follows (you may also wish to specify different output file names for each configuration as otherwise the results would be overwritten):

      |     Case    |   -l    |     -s      |  -p   |
      | ----------- | ------- | ----------- | ----- |
      | `Realistic` | 60 5400 | 13 15 17 10 | true  |
      | `Reduced`   | 60 100  | 13 15 17 10 | true  |
      | `Uniform`   | 60 100  | 13 15 17 10 | unif  |
      | `Dirichlet` | 60 100  | 13 15 17 10 | 10    |
      | `Biased`    | 80 100  | 1  10 1  10 | true  |
      | `Bias&Unif` | 80 100  | 1  10 1  10 | unif  |


2. Visualise the Results in Tabular Format: again, the default configuration (up to specification of which result files to use) is enough, but is given here for posterity:
   ```bash
   python Visualise_Predictions.py -r ../data/Compare_DS.npz ../data/Compare_ISAR.npz -s I II III IV
   ```
   

### Parameter Recovery from Synthethic Data (Section 5.4)

We replicate here the results using both the 'extreme' One-v-Rest schemas, as well as the MRC-Harwell Schemas.

#### One-vs-Rest Schemas

This is again a two-stage process. To replicate Fig. 4 (a):

1. Train the Model using the single-schema per-sample setting:
   ```bash
   python Simulate_Parameter_Learning.py -o ../data/Parameters_ISAR -r 0 -n 0 20 -l 500 100 -s 7 6 -i 0.001 0.005 0.01 0.05 0.1 0.5 1.0 -e
   ```
2. Visualise the results:
   ```bash
   python Visualise_Parameter_Learning.py -r ../data/Parameters_ISAR.npz
   ```
   
For Fig. 4 (b) follow the same procedure but we allow for a different schema per sample by passing the `-d` flag:
   ```bash
   python Simulate_Parameter_Learning.py -o ../data/Parameters_ISAR -r 0 -n 0 20 -l 500 100 -s 7 6 -i 0.001 0.005 0.01 0.05 0.1 0.5 1.0 -e -d
   ```
Following this, you can visualise results as before.

#### MRC Harwell Schemas

To replicate Fig. 5, we reuse the same scripts as above but with a different configuration (this is also the default configuration):

1. Train the Model under realistic conditions:
   ```bash
   python Simulate_Parameter_Learning.py -o ../data/Parameters_ISAR -r 0 -n 0 20 -l 60 5400 -s 13 11 -i 0.001 0.005 0.01 0.05 0.1 0.5 1.0 
   ```
   
2. Visualise the results: in this case, it is also helpful to visualise per-annotator Psi emissions (Fig. 5 (b)) using the `-i` flag
   ```bash
   python Visualise_Parameter_Learning.py -r ../data/Parameters_ISAR.npz -i
   ```
   
### Analysis of Mutual Information (Section 5.5)

This is a one-part script which will generate all the values in Table 3 (and more):

```bash
python Analse_MutualInformation.py
```


## Troubleshooting

* PROBLEM: Running the scripts gives me a `ImportError: cannot import name 'XXXXX'`.

  FIX: Make sure you are running the scripts from the actual directory in which it exists. This is due to the fact that the library is not installed, and hence, all tools are relative to the script location.
  
  
* PROBLEM: The script runs successfully, but fails when storing results with `FileNotFoundError: [Errno 2] No such file or directory: "XXXXXXX"`

  FIX: On Windows, you must replace forward slashes in the path with back-slashes!
  


