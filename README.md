
# Hero recognition

## **Overview**
Hero recognition is a classification model to classify the hero of Wild Rift in game message bar.
Currently, we only support x64Linux.
Read **MainGames-Report.pdf** for more infomation

## **1. Environment Setup**
### 1.1 Python

Tested on Python 3.8

### 1.2 Required Environment

Install all required environment:
```
pip install -r requirements.txt
```

## **2. Inference**

```
python3 infer.py --path path_to_image_folder --model big/small --device cuda/cpu 
```
