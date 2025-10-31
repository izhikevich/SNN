**SNN Transformer** 
by Eugene Izhikevich, SpikeCore.com
Patent pending 

**Data for training**

Use any sufficiently large text file. For example, I use 
https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

Change the name of the data file in line 555, if needed. 


**Compile and Run**

gcc -O3 -o main main.c 
./main

Loss values during training are saved in the FILE_NAME in line 20

**Deviations from the model in the paper**

1. Positional encoders PE are different for different look-up tables; see void cache_PE_index()

2. Positional encoders do not use anchors neurons; indexes are formed using element-wise comparisons with 0. Hence, the number of comparisons is POSITIONAL_DIM (parameter p in the paper)

3. Anchor neurons for z_i and z_j are the same, so their indexes are re-used by the V-index cache
