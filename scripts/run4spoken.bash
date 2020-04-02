#!/bin/sh
python cLL-ML.py --resDir results_spoken_stem_1 --pre confFiles/spoken/spoken_4_stemmed_stop.conf --cat all
python cLL-ML.py --resDir results_spoken_stem_2 --pre confFiles/spoken/spoken_4_stemmed_stop.conf --cat all
python cLL-ML.py --resDir results_spoken_stem_3 --pre confFiles/spoken/spoken_4_stemmed_stop.conf --cat all

for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_1/NoOfDataPoints object object confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem.csv; done
for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_2/NoOfDataPoints object object confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem.csv; done
for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_3/NoOfDataPoints object object confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem.csv; done


for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_1/NoOfDataPoints rgb rgb confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem_rgb.csv; done
for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_2/NoOfDataPoints rgb rgb confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem_rgb.csv; done
for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_3/NoOfDataPoints rgb rgb confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem_rgb.csv; done

for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_1/NoOfDataPoints shape shape confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem_shape.csv; done
for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_2/NoOfDataPoints shape shape confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem_shape.csv; done
for i in {1..10}; do python Validation/macro-pos5DescrNegDocVecdistractorTest.py results_spoken_stem_3/NoOfDataPoints shape shape confFiles/spoken/spoken_4_stemmed_stop.conf | tee -a eng_lem_shape.csv; done
