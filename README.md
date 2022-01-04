# 02456 2021 Project: Insect classification

- Important! The code can not run unless FaunaPhotonics has agreed for one to have access to their data, database, and code. However, we have run this notebook and saved the outputs for inspection.

## Datasets
Data is provided by FaunaPhotonics and can not be shared, as it is considered owned by the company and confidential.

## References
The python module `evex_scout` imported in labelled_data.generate_data is owned by FaunaPhotonics. The code is used to collect the start and end times of individual insect events from a raw data file. A module named `fpmodules` is used to collect data from a database, which is also owned by FaunaPhotonics. The database consists of labels of whether the extracted events from the raw data file is insect or not. All resources from Faunaphotonics is only used for creating labelled datasets.

The `EarlyStopping` class from machine_learning.early_stopping is from https://github.com/Bjarten/early-stopping-pytorch.

All other code in this repository is made by Freja Thoresen (S213769), Pascal Dufour (S011602), Nikolaj Sheller (C971666).

## Code
The labelled_data folder contains all code to create labelled data files. The file to run the code is make_data.py.
The machine_learning folder contains the model definition and training and validation loops, plus some other helper functions.
