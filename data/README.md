There are three example data files. Each file should follow the specific structure mentioned below:

1. example_train.csv
     - This file should be used when training each individual genomic prediction model
     - The format should be n_train x m+2
           - n_train: the total number of RILs in the training data
           - m+2: one id column, m markers and one phenotype column

2. example_test.csv
     - This file should be used when evaluating each individual genomic prediction model
     - The format should be n_test x m+2
           - n_test: the total number of RILs in the test data
           - m+2: one id column, m markers and one phenotype column

3. example_matrix.csv
     - This file should be used when evaluating the ensemble model
     - The format should be n_test x 8
           - n_test: the total number of RILs in the test data
           - 8: one id column, six predicted phenotype columns from each individual genomic prediction model and one phenotype column