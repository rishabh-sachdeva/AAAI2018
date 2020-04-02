#!/bin/sh
conf_file="conf_speech/audio_descriptions.csv_lemmed_stop.conf"
result_dir="results_speech_lem"
validate_file="speech_lem.csv"
echo $conf_file
echo $result_dir
#python cLL-ML.py --resDir $result_dir --cat object --pre $conf_file
for i in {1..3}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py $result_dir/NoOfDataPoints object object $conf_file | tee -a $validate_file; done
#python process_output.py $validate_file