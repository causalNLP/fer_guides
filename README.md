# 1. Tips for Running LLMs on Euler

## Location of the models
Nowadays all new models are in HuggingFace Hub. Originally Llama weights were requested to Meta, so I stored the weights converted to HuggingFace format in the following paths:

Llama 7B: 

Alpaca 7B:

For loading new models I recommend to set the cache directory to your scratch folder, every Euler user has a scratch folder at "/cluster/scratch/{username}":

## Models smaller or equal to 13B
Getting resources in Euler for this model size is easy. 

I use a python script that reads a csv with the prompts and write the results in another csv. You can find an example here: [script](./examples/script.py).

Then I submit a job to the cluster with a script like the following:  [job](./examples/job.sh)

To submit the job run the following command:
```
sbatch < job.sh
```

- Resources: 
For 7B models I usually request 2 "rtx_2080_ti" GPUs, by setting the parameter device="auto" the transformers library automatically uses all GPUs available.

For larger models you just have to either increase the number of requested GPUs or request a bigger one. However, sometimes there are very long queues when you make those type of requests. Just request GPUS that you need so you don't block other users' jobs.


## Models larger than 13B
It usually takes very long to get enough resources to run these models. In these cases I recommend to use quantization. 
Some people has uploaded quantized versions of models to HuggingFace Hub. E.g. https://huggingface.co/TheBloke

For Llama models follow the instructions in this repo: https://github.com/qwopqwop200/GPTQ-for-LLaMa
For the 65B model, in addition I use the llama_inference_offload.py script which loads just some layers of the model at a time to the GPU. 



# 2. Mechanistic Interpretability

## ROME
https://github.com/kmeng01/rome/tree/main


Useful library with causal tracing examples. They have a Colab notebook with examples:
https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/causal_trace.ipynb
I adapted some functions of the library in the following [script](./examples/tracing.py) to work with Llama, in specific the function "layername".
The example has other adaptaions for the specific use case, i.e. tracking the probabilities of specific tokens, but the main modification to make it work with Llama is the layername function.

## TransformerLens
https://github.com/neelnanda-io/TransformerLens/tree/main
Very useful library with demos that facilitate the use of LLMs for interpretability. It has examples for activation patching and logit attribution.

