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

1. Run a `git pull` on this repository to get the latest commits

2. Run `conda env create -f environment.yml`

3. Run `conda activate datares1-env`

4. Run `jupyter notebook` or run `jupyter lab`
