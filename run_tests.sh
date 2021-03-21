#!/bin/sh

model_configs_file=configs_list.txt

while read line; do
  python run.py -c ./configs/$line
done < $model_configs_file
