# Robust Bias Detection in MLMs and its Application to Human Trait Ratings
*To appear at NAACL 2025*

## Templates, Attributes and Targets

**Character Traits**

factor1: empathy, factor2: order, factor3: resourceful, factor4: serenity

**Personality Traits**

factor1: extroversion, factor2: agreeableness, factor3: conscientiousness, factor4: emotional stability, factor5: openness

Refer to `data/`

## Bias detection in MLMs

Run `evaluate_bias_mlm.sh` inside `code/mlm/`

## Effect of Negative Traits

Run `effect_of_negative_traits_roberta.sh` inside `code/eval_roberta_negative_traits/`

## Bias detection in llama3

Run `evalauate_bias_llama3.sh` inside `code/alm/`

## MLM bias detection using crowdsourced dataset (CrowS-Pairs)

Run `evaluate_bias_cps.sh` inside `code/crowspairs/code/`

## Bias mitigation in MLMs

Run `debias_roberta.py` inside `code/bias_mitigation/`