# ISAR (Inter-Schema AdapteR)

This repository contains python code for replicating the Results in the Paper:
> "A Model for Learning Across Related Label Spaces", Under Review

**Due to the nature of our collaboration with MRC Harwell, we can only provide the data (and code) for the simulation components of the study.**

## Contents

1. Repository Structure
1. Installation and Setting Up
1. Reproducing the Experiments

## Repository Structure

At the top level, there are two `requirements` files, for easy installation of the necessary packages. All the code is packaged under the `python` directory which contains the following packages:
 * **Tools**: Contains in-house libraries/modules used by the remainder of the scripts: this includes a multi-processing wrapper for exploiting multi-core architectures and extensions to some numpy methods.
 * **Models**: Contains the ISAR implementations for the Multinomial Class-Conditional and Inter-Annotator Variability models.
 * **20 Newsgroups**: Contains scripts for replicating the results of the simulations on the 20 Newsgroups Dataset (i.e. Section 5.1 in the Paper).
 * **MRC Harwell**: Contains scripts for replicating the results of the simulations and analysis on the MRC Harwell Data (i.e. Sections 5.2.2, 5.3 and 5.4).

There is also a `data` directory which at the outset contains the parameters for the MRC-Harwell-trained models which form the basis of our simulations. This is also the default directory for storing results from simulation runs (but can be configured, see below).
 
In general, the replication scripts are split into two stages: one for simulating and training the models, and the other for visualising results.

## Installation and Setting Up

The repository is designed to be a self-contained implementation, subject to some standard library requirements

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

That is all! You are set to go.

## Replicating the Results

All scripts below can be run from the command line by navigating to the directory containing the appropriate script and running it through python:

```bash
python <SCRIPT_NAME.py> [COMMAND LINE ARGUMENTS]
```

All the scripts support some form of tweaking by way of command line arguments passed through as key-value pairs. Typically, the default values generate the figures/tables exactly as they appear in the paper: where this needs to be run multiple times with different configurations, the settings for each run are provided. However, feel free to experiment with the values as this can help identify interesting avenues. Calling the script with the `-h` key generates a helpful summary of what each parameter controls.

We will now describe replication of the results in each relevant section

### Predictive Performance on the 20 Newsgroups Data-set (Section 5.1)

Replicating Fig. 5 is a three-stage process.

1. Load the 20 Newsgroups Data-set and format it accordingly to our problem. The below configuration is also the default and is only specified here for completeness' sake.
   ```
   python Load.py -o ../../data/20NG_BOW -r 1010 -f 5000 -t 0.2
   ```
2. Train the Model(s). This has to be called 4 times to generate the 4 plots in Fig. 5, each time changing the bootstrap percentage (and the output file name, otherwise it will just overwrite the other results):
   ```
   python Learn.py -s ../../data/20NG_BOW.npz -o ../../data/<NAME> -r 0 -n 0 30 -f <FRACTION> -a 0.005
   ```
   where `<FRACTION>` should be one from [0.02, 0.05, 0.1, 0.2]. The default is for 0.02.
   
3. Visualise the Results. This is called once, passing in as a list the (4) result files. Assuming each file was named Results_{} where {} is a standin for the percentage, this can be by achieved:
   ```
   python Visualise.py -r ../../data/Results_02.npz ../../data/Results_05.npz ../../data/Results_10.npz ../../data/Results_20.npz -n 2% 5% 10% 20%
   ```
   The second parameter allows customisation of the naming of the lines on the plot.


### Analysis on the Crowdsourced Characterisation Task (Section 5.2)

Due to the ownership of the data, we can only show here the results for the inference of the latent state on simulated 
data (i.e. Section 5.2.2). That being said, care was taken to model the data generation process as close as possible. 

Tables 4/6 can be replicated in a two-step process.

1. Train the DS/ISAR Models on the entire dataset. This will generate two files, one each for DS and ISAR: just pass `none` to the -o parameter if you wish to suppress any simulation component. The below is the code to generate the results for the `Realistic` case (which is also the default): note that this will take about 10 Hours since it has not **yet** been optimised to run multiple runs
in parallel.
   ```bash
   python Learn_Compare.py -o ../../data/Compare_DS ../../data/Compare_ISAR -r 0 -n 0 20 -l 60 5400 -f 10 -s 13 15 17 10
   ```
   In all cases, the following parameters are common:
    * `-r 0` (Random State)
    * `-n 0 20` (Start index and number of independent runs)
    * `-f 10` (Number of folds)
 The remaining settings for each run are as follows (you may also wish to specify different output file names for each configuration as otherwise the results would be overwritten):

  |     Case    |   -l    |     -s      |  -p   |
  | ----------- | ------- | ----------- | ----- |
  | `Realistic` | 60 5400 | 13 15 17 10 | true  |
  | `Reduced`   | 500 100 | 13 15 17 10 | true  |
  | `Uniform`   | 500 100 | 13 15 17 10 | unif  |
  | `Dirichlet` | 500 100 | 13 15 17 10 | 10    |
  | `Biased`    | 500 100 | 1  10 1  10 | true  |
  | `Bias&Unif` | 500 100 | 1  10 1  10 | unif  |  


2. Visualise the Results in Tabular Format: again, the default configuration (up to specification of which result files to use) is enough, but is given here for posterity:
   ```bash
   python Visualise_Compare.py -r ../../data/Compare_DS.npz ../../data/Compare_ISAR.npz -s I II III IV
   ```


### Parameter Estimation in the multi-annotator scenario (Section 5.3)

We replicate here the results using both the 'extreme' One-v-Rest schemas, as well as the MRC-Harwell Schemas.

#### One-vs-Rest Schemas (Section 5.3.1)

This is again a two-stage process. To replicate Fig. 6 (a):

1. Train the Model using the single-schema per-sample setting:
   ```bash
   python Learn_Parameters.py -o ../../data/Parameters_ISAR -r 0 -n 0 20 -l 500 100 -s 7 6 -i 0.001 0.005 0.01 0.05 0.1 0.5 1.0 -e
   ```
2. Visualise the results:
   ```bash
   python Visualise_Parameters.py -r ../../data/Parameters_ISAR.npz
   ```
   
For Fig. 6 (b) follow the same procedure but we allow for a different schema per sample by passing the `-f` flag:
   ```bash
   python Learn_Parameters.py -o ../../data/Parameters_ISAR -r 0 -n 0 20 -l 500 100 -s 7 6 -i 0.001 0.005 0.01 0.05 0.1 0.5 1.0 -e -f
   ```
Following this, you can visualise results as before.

#### MRC Harwell Schemas (Section 5.3.2)

To replicate Fig. 8, we reuse the same scripts as above but with a different configuration (this is also the default configuration):

1. Train the Model under realistic conditions:
   ```bash
   python Learn_Parameters.py -o ../../data/Parameters_ISAR -r 0 -n 0 20 -l 60 5400 -s 13 11 -i 0.001 0.005 0.01 0.05 0.1 0.5 1.0 
   ```
   
2. Visualise the results: in this case, it is also helpful to visualise per-annotator Psi emissions (Fig. 8 (b)) using the `-i` flag
   ```bash
   python Visualise_Parameters.py -r ../../data/Parameters_ISAR.npz -i
   ```
   
### Analysis of Mutual Information (Section 5.4)

This is a one-part script which will generate all the values in Table 5:

```bash
python AnalseMutualInformation.py
```



## Troubleshooting

* PROBLEM: Running the scripts gives me a `ImportError: cannot import name 'XXXXX'`.

  FIX: Make sure you are running the scripts from the actual directory in which it exists. This is due to the fact that the library is not installed, and hence, all tools are relative to the script location.
  
* PROBLEM: Executing '20 Newsgroups\Visualise.py' gives me a bunch of errors ending with `ValueError: unsupported format character ''' (0x27) at index 109`

  FIX: This appears to be a limitation with running the script on Windows, and the way it handles special characters (in this case the percentage sign). To avoid this, make sure you specify the names of the lines using the `-n` switch.
  
* PROBLEM: The script runs successfully, but fails when storing results with `FileNotFoundError: [Errno 2] No such file or directory: "XXXXXXX"`

  FIX: On Windows, you must replace forward slashes in the path with back-slashes!
  


