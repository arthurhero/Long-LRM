# Self-Reimplemented Version of Long-LRM  

[**Project Page**](https://arthurhero.github.io/projects/llrm/index.html)  

This repository contains a self-reimplemented version of **Long-LRM**, including the model code, as well as training and evaluation pipelines. The reimplemented version has been verified to match the performance of the original implementation.  

---

## Tentative TO-DO List

- [x] Sample config files
- [x] Script for converting raw DL3DV files into reuqired format
- [x] Config files for training on DL3DV
- [ ] 2D GS support
- [ ] Post-prediction optimization
- [ ] Pre-trained model weights (please do not count on this)

---

## Getting Started  

### 1. Prepare Your Data  
Format your dataset following the structure of the example data in `data/example_data`.  
- Each dataset should contain one `.txt` file which lists the paths to the JSON files for each scene, with one path per line.

### 2. Configure Your Model  
Create a config file in YAML format.  
- Include fields for `training`, `data`, and `model` settings.  
- You may also supply a *default* config file to `main.py`, fields of which will be overwritten by the conflicting  values in the custom config file. This is handy for running multiple experiments with only a few config changes.

### 3. Train or Evaluate the Model  
Run `sh create_env.sh` to install the required packages.
Use `torchrun` to launch the training loop:  
```bash  
torchrun --nproc_per_node $NUM_NODE --nnodes 1 \
         --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint localhost:$PORT \
         main.py --config path_to_your_config.yaml \
         --default-config path_to_your_default_config.yaml
```  

#### Switch to Evaluation Mode  
To run the evaluation loop, add the `--evaluation` flag to the command line:  
```bash  
torchrun --nproc_per_node $NUM_NODE --nnodes 1 \
         --rdzv_id $JOB_ID --rdzv_backend c10d --rdzv_endpoint localhost:$PORT \
         main.py --config path_to_your_config.yaml \
         --default-config path_to_your_default_config.yaml \
         --evaluation
```  

---