#!/bin/sh
conf_file="conf_speech/audio_descriptions.csv_stemmed_stop.conf"
result_dir="results_speech_stem"
validate_file="speech_stem.csv"
echo $conf_file
echo $result_dir
#python cLL-ML.py --resDir $result_dir --cat object --pre $conf_file
for i in {1..3}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py $result_dir/NoOfDataPoints object object $conf_file | tee -a $validate_file; done
#python process_output.py $validate_file