#!/bin/bash

# llama3_association_score.py
# For details on available arguments and their usage, refer to the parse_argument() method.

# sentence_pair_generation.py
# For details on available arguments and their usage, refer to the parse_argument() method.

# generate_pairwise_association_scores.py
# For details on the command-line arguments, see the argument parsing section in the script.

# Association scores for neo-pronouns (e.g., charater trait: empathy (factor1))
python3 llama3_association_score.py \
            --neo_attr_path ../../data/attributes/neo.txt \
            --templates_path ../../data/templates.txt \
            --target_path ../../data/character_traits/positive/factor1.txt \
            --output_path ../../output/alm_output/factor1_paired_true_cases_neo.tsv 

# Association scores for binary gender (e.g., charater trait: empathy (factor1))
python3 llama3_association_score.py \
            --male_attr_path ../../data/attributes/males.txt \
            --female_attr_path ../../data/attributes/females.txt \
            --templates_path ../../data/templates.txt \
            --target_path ../../data/character_traits/positive/factor1.txt \
            --output_path ../../output/alm_output/factor1_binary.tsv 

# "sentence generation from templates" (balance determiner selection)
# e.g., 'my father' and 'your mother', we add the alternatives 'your father' and 'my mother' for balance.
python3 ../sentence_pair_generation.py \
            --male_attr_path ../../data/attributes/males.txt \
            --female_attr_path ../../data/attributes/females.txt \
            --in_file_path ../../output/alm_output/factor1_binary.tsv  \
            --output_path ../../output/alm_output/factor1_paired_true_cases_binary.tsv 

# generate pairwise dataset for analyzing pairwise gender bias (M-F, M-N, F-N)
python3 ../generate_pairwise_association_scores.py \
        --binary_filepath ../../output/alm_output/factor1_paired_true_cases_binary.tsv \
        --neo_filepath ../../output/alm_output/factor1_paired_true_cases_neo.tsv \
        --output_dir ../../output/alm_output/

# Example to analyze bias using model_lme.R (bias score and R2 effect size)
# (obtain bias score and effect size between female and neo - change the file accordingly (female_neo, male_neo, male_female))

declare -A r_models
r_models["model_lme"]="model_lme.R"

run(){
    local selection_criteria=$1
    /bin/echo
    
    # character traits (1..4) [4 trait dimensions]
    # personality traits (1..5) [5 trait dimensions]
    for factor_num in {1..4}
    do
        # iterate over r_models dictionary
        for r_model in "${!r_models[@]}"; do
            echo "selection:${selection_criteria}, factor:${factor_num}"
            
            Rscript ../${r_models[$r_model]} \
                    --selection "${selection_criteria}" \
                    --factor "factor${factor_num}" \
                    --input_file "../output/alm_output/factor${factor_num}_paired_true_cases_female_neo.tsv" \
                    --output_txt_file "../output/alm_output/bias_R2_factor${factor_num}_${selection_criteria}_female_neo.txt" \
                    --output_tsv_file "../output/alm_output/bias_R2_factor${factor_num}_${selection_criteria}_female_neo.tsv"   
        done
    done
}

# run using all templates
# options: "t1_t2", "t3_t4", "t1_to_t4", "t3_to_t6", "t1_to_t6"
run "t1_to_t6"