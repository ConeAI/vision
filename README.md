# vision
Core vision code

# Deeppose
Based on https://github.com/samitok/deeppose and http://arxiv.org/abs/1312.4659

## Runbook
To generate data:

```
python3 deeppose_data.py --data_dir=/Users/wojtekswiderski/Documents/GitHub/ConeAI/vision/data/deeppose
```

To train model:

```
python3 deeppose_train.py --data_dir ~/documents/github/ConeAI/vision/data/deeppose/ --batch_size 32
```
