import pandas as pd

## Read your genetic marker data 
## n x m: n is the total number of lines and m is the total number of markers
data = pd.read_csv("YOUR GEMETIC MARKER DATA")

#===== PED =====#
## Split alleles into two
data_splited = []
for i in range(data.shape[0]):
     data_splited += [[x for x in ''.join(data.iloc[i,:].tolist())]]
data_splited = pd.DataFrame(data_splited)
data_splited.columns = [ele for ele in list(data.columns) for i in range(2)]
 
## Generate information for ped file
FID = ['FID'] * data.shape[0]
ID = list(range(1,data.shape[0]+1))
PID = [0] * data.shape[0]
MID = [0] * data.shape[0]
S = [0] * data.shape[0]
AF = [0] * data.shape[0]

info = pd.DataFrame({'Fam_ID':FID, 'Sam_ID':ID, 
                     'P_ID':PID, 'M_ID':MID, 
                     'Sex':S, 'Affection':AF})

## Create the data into ped format
data_ped = pd.concat([info,data_splited], axis=1)
data_ped.to_csv('PED FILE.ped', header=False, index=False, sep ='\t')

#=== MAP =====#
## Generate information for map file
chromosome = pd.Series(list(data.columns)).str.split("_", n = 1, expand = True)

CHROM = chromosome.iloc[:,0].to_list()
ID = list(data.columns)
DIS =  [0] * data.shape[1]
LOC = chromosome.iloc[:,1].to_list()

## Create the data into map format
data_map = pd.DataFrame({'Chromosome':CHROM, 'ID':ID, 'G_distance':DIS, 'P_position':LOC})
data_map.to_csv('MAP FILE.map', header=False, index=False, sep ='\t')