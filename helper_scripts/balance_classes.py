#!/usr/bin/env python

import glob
import shutil
import uuid
import os
import random
import sys

INPUT_DIRECTORY = "/mount/SDC/casava/split-data/output/train"
OUTPUT_DIRECTORY = "/mount/SDC/casava/split-data/output-normalized/train"

file_sizes = {}
files = {}
for _class in os.listdir(INPUT_DIRECTORY):
    files[_class] = glob.glob(os.path.join(INPUT_DIRECTORY, _class, "*"))
    file_sizes[_class] = len(files[_class])

_keys = file_sizes.keys()
_class_sizes = [file_sizes[x] for x in _keys]
_max_class_size = max(_class_sizes)
normalization_factor = [_max_class_size - file_sizes[x] for x in _keys]

print normalization_factor

for _idx, _class in enumerate(_keys):
    #Copy over the old folder
    print "Copying the images for the class : ", _class, "Size : ", _class_sizes[_idx]
    shutil.copytree(os.path.join(INPUT_DIRECTORY, _class), os.path.join(OUTPUT_DIRECTORY, _class))
    print "Normalizing class size by duplicating %d images...." % (normalization_factor[_idx])
    for k in range(normalization_factor[_idx]):
        _image_path = random.choice(files[_class])
        target_filename = "DUPLICATE_"+str(uuid.uuid4())+".JPG"
        target_path = os.path.join(OUTPUT_DIRECTORY, _class, target_filename)
        shutil.copy(_image_path, target_path)
        print "\rCopied %d files" % k,
        sys.stdout.flush()

    print ""
