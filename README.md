**SNN Transformer** <br />
Eugene Izhikevich (2025) Spiking Manifesto, https://arxiv.org/pdf/2512.11843 <br />
Hardware patent pending 

**Data for training**

Use any sufficiently large text file. Provide filename in line 546


**Compile and Run**

```ps1
gcc -O3 -o main main.c -lm
./main
```

Validation loss values during training are saved in the FILE_NAME in line 16

**Deviations from the model in the paper**

1. Positional encoders PE are different for different look-up tables; see void cache_PE_index()

2. Positional encoders do not use anchors neurons; indexes are formed using element-wise comparisons with 0. Hence, the number of comparisons is POSITIONAL_DIM (parameter p in the paper)

3. Anchor neurons for z_i and z_j are the same, so their indexes are re-used by the V-index cache

4. The learning of token_embedder is disabled. It does not improve performance

**Commands**

Resume training from a checkpoint:
```ps1
./main --resume model_50000.bin
```

Inference prompt generation:
```ps1
./main model_50000.bin "Go forth my prince" 200    # custom length
```