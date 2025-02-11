#!/bin/bash

# Effect of negative traits (Section 3.2.4.1) [RoBERTa-large]

# Association scores for binary gender (e.g., charater trait: empathy (factor1) [positive traits])
python3 ../mlm/mlm_association_score.py \
        --model roberta-large \
        --male_attr_path ../../data/attributes/males.txt \
        --female_attr_path ../../data/attributes/females.txt \
        --templates_path ../../data/templates.txt \
        --target_path ../../data/character_traits/positive/factor1.txt \
        --output_path ../../output/mlm_output/factor1_binary_positive.tsv 

# Association scores for binary gender (e.g., charater trait: empathy (factor1) [negative traits])
python3   ../mlm/mlm_association_score.py \
        --model roberta-large \
        --male_attr_path ../../data/attributes/males.txt \
        --female_attr_path ../../data/attributes/females.txt \
        --templates_path ../../data/templates_negative_traits.txt \
        --target_path ../../data/character_traits/negative/factor1.txt \
        --output_path ../../output/mlm_output/factor1_binary_negative.tsv 

# "sentence generation from templates" (balance determiner selection) [positive traits]
# e.g., 'my father' and 'your mother', we add the alternatives 'your father' and 'my mother' for balance. 
python3 ../sentence_pair_generation.py \
        --male_attr_path ../../data/attributes/males.txt \
        --female_attr_path ../../data/attributes/females.txt \
        --in_file_path ../../output/mlm_output/factor1_binary_positive.tsv  \
        --output_path ../../output/mlm_output/factor1_paired_true_cases_binary_positive.tsv 

# [negative traits]
python3 ../sentence_pair_generation.py \
        --male_attr_path ../../data/attributes/males.txt \
        --female_attr_path ../../data/attributes/females.txt \
        --in_file_path ../../output/mlm_output/factor1_binary_negative.tsv  \
        --output_path ../../output/mlm_output/factor1_paired_true_cases_binary_negative.tsv 

# combine results of both positive and negative traits (e.g., charater trait: empathy (factor1))
python3 combine_positive_negative_traits_results.py \
        --pos_trait_path ../../output/mlm_output/factor1_paired_true_cases_binary_positive.tsv \
        --neg_trait_path ../../output/mlm_output/factor1_paired_true_cases_binary_negative.tsv \
        --output_path ../../output/mlm_output/factor1_paired_true_cases_binary_pos_neg.tsv 


# Run model_lme.R
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
                    --input_file "../output/mlm_output/factor${factor_num}_paired_true_cases_binary_pos_neg.tsv" \
                    --output_txt_file "../output/mlm_output/bias_R2_factor${factor_num}_${selection_criteria}_binary.txt" \
                    --output_tsv_file "../output/mlm_output/bias_R2_factor${factor_num}_${selection_criteria}_binary.tsv"   
        done
    done
}

# run using all templates
# options: "t1_t2", "t3_t4", "t1_to_t4", "t3_to_t6", "t1_to_t6"
run "t1_to_t6"