#!/bin/sh
conf_file="conf_text/text_desciptions.csv_stop_raw.conf"
result_dir="results_text_stop"
validate_file="text_stop.csv"
echo $conf_file
echo $result_dir
#python cLL-ML.py --resDir $result_dir --cat object --pre $conf_file
for i in {1..3}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py $result_dir/NoOfDataPoints object object $conf_file | tee -a $validate_file; done
#python process_output.py $validate_file