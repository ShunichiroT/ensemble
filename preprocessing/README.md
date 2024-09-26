# Preprocessing

Genetic markers are preprocessed by linkage-disequilibrium (LD) filtering from PLINK 1.9.

1) Data containing the genetic markers is assumed to be in both ped and map formats
2) The data is converted into a bed file
3) Unfiltered markers are extracted from the LD-calculation
4) Filtered markers are removed from the converted bed file
5) The filtered bed file is recoded into ped and map files
