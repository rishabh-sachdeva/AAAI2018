#!/bin/sh
conf_file="conf_text/text_desciptions.csv_stemmed_stop.conf"
result_dir="results_text_stem"
validate_file="text_stop.csv"
echo $conf_file
echo $result_dir
python cLL-ML.py --resDir $result_dir --cat object --pre $conf_file
for i in {1..3}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py result_dir/NoOfDataPoints object object $conf_file | tee -a $validate_file; done