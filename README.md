# ConsistentID: Portrait Generation with Multimodal Fine-Grained Identity Preserving

This repository contains the implementation of the paper.
Anonymous demo: http://consistentid.natapp1.cc/

![IMGs_v2](https://github.com/user-attachments/assets/8e74a2e6-f82e-4969-87bd-b5371cb20679)


## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Train

```setup
bash train_bash.sh
```


## Infer

```setup
python infer_demo/infer.py
```

## Infer with ControlNet

```setup
python -m demo.inpaint_demo
python -m demo.controlnet_demo
```

