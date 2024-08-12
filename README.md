# ConsistentID: Portrait Generation with Multimodal Fine-Grained Identity Preserving

This repository contains the implementation of the paper.
Anonymous demo: http://consistentid.natapp1.cc/
![Framework](https://github.com/user-attachments/assets/1b4db078-a269-4119-88e6-f522aa6341b1)


![IMGs_v2](https://github.com/user-attachments/assets/1cc66d57-aecd-4995-b511-a0f86d7940a1)


![IMGs_facial](https://github.com/user-attachments/assets/591aee2f-60a8-4df3-86ff-0afa226c89c3)


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
python infer_controlnet_demo/inpaint_demo.py
python infer_controlnet_demo/controlnet_demo.py
```

