# A Survey Comparing xLSTM and Transformer Architectures

## Abstract

LSTMs were the first major breakthrough in language modeling. However, due to certain limitations (which will be discussed further), transformers took over. But what if we overcome these limitations?
Can LSTMs perform better if we scale them to the level of contemporary transformer architectures? And does architecture even matter, or is it all about the number of parameters in the model?


# Reproduce experiments

Set up the environment:

```bash
conda env create -n xlstm -f environment.yaml 
conda activate xlstm
pip install xlstm llm-trainer
```


# References

Maximilian Beck, Korbinian Pöppel, et al. xLSTM: Extended Long Short-Term Memory. ArXiv, 2405.04517, 2024.