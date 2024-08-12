# ConsistentID: Portrait Generation with Multimodal Fine-Grained Identity Preserving

This repository contains the implementation of the paper.
Anonymous demo: http://consistentid.natapp1.cc/
![Framework](https://github.com/user-attachments/assets/1b4db078-a269-4119-88e6-f522aa6341b1)


![IMGs_v2](https://github.com/user-attachments/assets/73ced00b-074c-42c5-a245-882d0b52e48f)



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

