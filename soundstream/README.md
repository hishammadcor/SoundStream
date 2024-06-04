#SoundStream
Forja Labs Audio Engineer Recruitment Challenge


significant
- Implemented using pytorch-lightning.
- You can install the necessary libraries using the method below.
```
poetry shell
poetry update
```
- Learning can be performed with the code below.
```
python main fit --config/base.yaml
```
- Hyperparameters can be modified in config/base.yaml.
- The Inference example can be run with the code below.
```
python infer.py "checkpoint path" "inference data"
```