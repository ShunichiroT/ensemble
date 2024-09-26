# Preprocessing

It consists of imputation and linkage-disequilibrium(LD)-pruning

Imputation
1) Prepare the original genetic marker data
2) Remove markers with more missing values than the threshold
3) Calculate and replace the rest of the missing values with the most frequent allele in each marker

LD-pruning
1) Assume that the genetic markers are in both ped and map formats. PLINK 1.9 is installed
2) Convert the data into a bed file
3) Extract Unfiltered markers from the LD-calculation
4) Remove filtered markers from the converted bed file
5) Recode the filtered bed file into ped and map files
