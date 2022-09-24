# Pytorch-RAF-Resnet
A Pytorch Implementation of Facial Expression Recognition On RAF Dataset based on Resnet, I got 86.75% percision on 2 GPUs, you may get better results if you  want to try more pretrained models.

## Denpendency

```
pip install -r requirements.txt
```

## Data

- [Download link](http://www.whdeng.cn/raf/model1.html#dataset)

```
datasets/
└── raf-basic
    ├── EmoLabel
    │   └── list_patition_label.txt
    └── Image
        └── aligned
```

`list_patition_label.txt` data format is:

```
train_00001.jpg 5
train_00002.jpg 5
train_00003.jpg 4
train_00004.jpg 4
......
```
And the images are in aligned directory.




## Training

```
python mainpro_RAF.py
```


## Reference

[https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)
