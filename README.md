# Research-Bipartite-Matching-Problem

### To do: Maximal Matching Algorithm

Build a bipartite matching algorithm in python -- maximal bipartite matching. 

Steps: 

1. Define a bipartite graph structure

2. 1 set of keys from table a. one set of keys from table b. 

Notes: 

- The other set would describe the relationship between the tables (weighted relationship). 

- Any possible match can be an edge in this graph

## Tasks after Meeting 03.25.2020

### Generalize the code to 

1. Allow for an arbitrary weight function to handle both maximum and minimum 

2. Allow to be given two dataframes, and a function that gives the correspondence


## Tasks after Meeting 04.01.2020

### Use the bipartite matching algorithm on the DBLP-ACM dataset to:

1. See areas where we can further generalize the code for bipartite matching

2. Check the accuracy performance of the bipartite matching (using the ground truth dataset)

Note: The dataset is not added to this repository due to memory constraints. Please put the dataset in the same folder as the notebook file while running the code. 

Link for the dataset: https://www.openicpsr.org/openicpsr/project/100843/version/V2/view

## Tasks after Meeting 04.10.2020

### Generalize the code to:

1. Use the "_1" and "_2" labels so that the code can be further generalized to different matchings

2. Use a built in similarity metric library written in source C/C++ so that we can eliminate any slowness stemming from the similarity metrics that I've handcoded myself.

### Update: Anaconda Compability

* Jupyter did not recognize some of the already installed libraries (the libraries that I tried to install through PyPI's similarity metric packages). Running jupyter in a conda environment solved the problem. In order to run this jupyter notebook in a conda environment, follow these steps:

1. Make sure you have anaconda installed https://docs.continuum.io/anaconda/install/#

2. Run a `git pull` on this repository to get the latest commits

3. Run `conda env create -f environment.yml`

4. Run `conda activate datares1-env`

4. Run `jupyter notebook` or run `jupyter lab`



## Tasks after Meeting 04.27.2020
### Modularize the code

The code structure is now as follows:

- `one_to_one.py` includes all the essential functions that are contained in the `1-1`bipartite matching algorithm

- Tests use the `one_to_one` module in the `src` folder. The tests can be found in the `tests/` folder

- There are 2 test files:

  - The `one_to_one_basic.py` file contains tests that check whether the functions in the core module are working correctly.

  - The `one_to_one_advanced.py` file contains tests that evaluate the precision and accuracy of the matchings according to the perfect mapping that is provided as an example in this file.


## Tasks after Meeting 05.08.2020
### Write a module for the `1-n` matching case.

### Process:

1. Duplicate the "1" table to have a user inputted amount of duplicates

2. Do the bipartite matching on that duplicated "1" table. The resulting output would look like this:

Resulting Matching Excerpt:

USA_1 ------> US

USA_2 ------> U.S.A

3. De-duplicate / Collapse the results to have a `1-n` matching that looks like follows:

Collapsing the duplicates:

USA ---> US, U.S.A

The updated code structure is now as follows:

- `one_to_n.py` includes all the essential functions that are contained in the `1-n`bipartite matching algorithm

- Tests use the `one_to_n` module in the `src` folder. The tests can be found in the `tests/` folder

- There are 2 test files:

  - The `one_to_n_basic.py` file contains tests that check whether the functions in the `1-n` module are working correctly.

  - The `one_to_n_advanced.py` file contains tests that evaluate the efficiency of the `1-n` matching for the large dataset example `DBLP-ACM` dataset.

# Tasks after Meeting 05.15.2020
## Investigate the Transitive Closure Problem for 1 Table.

- The source file for the work is aggragated in `src/transitive_closure.py` and the tests are aggragated in `tests/transitive_basic.py` and `tests/transitive_advanced.py`. The latter file is still in the works with the hope of finishing it once the problem is solved using the small scale data that covers all the edge cases. 

## Transitive Closure Problem in 1 Table

### Problem Statement: Assume that we have many of the same entities in different representations in the same column of the table. We want to find a way to indicate that they are the same entities by mapping them.

For example, assume the following table:

Table A  |

_________|

Countries|

_________|

USA

US

CHINA

CHNA

CANADA

The matching outcome we want is the following:

Table A       (Duplicated Table A)

USA ---------- US

US  ---------  USA

CHINA ------- CHNA 

CHNA -------- CHINA           

CANADA ---X-- CANADA
