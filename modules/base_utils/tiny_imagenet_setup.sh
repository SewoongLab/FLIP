wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/
python modules/base_utils/tiny_imagenet_fix_val.py