#!/bin/bash

categories_folder=('age' 'disability' 'gender' 'nationality' 'physical_appearance' 'race_color' 'religion' 'sexual_orientation' 'socioeconomic')

categories=('age' 'disability' 'gender' 'nationality' 'physical-appearance' 'race-color' 'religion' 'sexual-orientation' 'socioeconomic')

i=0

source activate py3_conda


python 1_extract_bias_type.py   --bias_type "${categories[$i]}" \
                                --input_file ../data/crows_pairs_anonymized.csv \
                                --output_file ../output/${categories_folder[$i]}/crowspair_${categories_folder[$i]}.tsv

python 2_crowspair_main_code.py --input_file ../output/${categories_folder[$i]}/crowspair_${categories_folder[$i]}.tsv \
                                --lm_model roberta \
                                --output_file ../output/${categories_folder[$i]}/original_out_${categories_folder[$i]}.tsv

python 3_extract_crowspair_assocation_score.py --input_file ../output/${categories_folder[$i]}/crowspair_${categories_folder[$i]}.tsv \
                                                --output_file ../output/${categories_folder[$i]}/crowpairs_${categories_folder[$i]}_scores.tsv

Rscript 4_bias_assessment.R --input_file ../output/${categories_folder[$i]}/crowpairs_${categories_folder[$i]}_scores_formatted.tsv > ../output/${categories_folder[$i]}/bias_analysis_${categories_folder[$i]}_output.txt 2>&1


# Example

# python 1_extract_bias_type.py   --bias_type gender \
#                                 --input_file ../data/crows_pairs_anonymized.csv \
#                                 --output_file ../output/gender/crowspair_gender.tsv

# python 2_crowspair_main_code.py --input_file ../output/gender/crowspair_gender.tsv \
#                                 --lm_model roberta \
#                                 --output_file ../output/gender/original_out_gender.tsv

# python 3_extract_crowspair_assocation_score.py --input_file ../output/gender/crowspair_gender.tsv \
#                                                 --output_file ../output/gender/crowpairs_gender_scores.tsv

# Rscript 4_bias_assessment.R --input_file ../output/gender/crowpairs_gender_scores_formatted.tsv > ../output/gender/bias_analysis_gender_output.txt
