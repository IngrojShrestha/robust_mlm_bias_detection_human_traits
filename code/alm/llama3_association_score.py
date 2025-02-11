import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime
import functools
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

print = functools.partial(print, flush=True)

access_token = "ACCESS_TOKEN"
model_id = "meta-llama/Meta-Llama-3.1-8B"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_vowel(word):
    import eng_to_ipa as ipa

    # CMU to IPA notations
    symbols = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
               "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
               "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ", "ow": "oʊ", "oy": "ɔɪ",
               "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}

    vowels = ["ə", "eɪ", "ɑ", "æ", "ə", "ɔ", "aʊ", "aɪ", "ɛ", "ər", "ɪ", "oʊ", "ɔɪ", "u", "i"]

    phoneme = ipa.convert(word)
    phoneme = phoneme.replace("ˈ", '').replace("ˌ", '')
    if phoneme[0] in vowels:
        return True
    else:
        return False

def print_args(args):
    
    print("=" * 100)
    
    if hasattr(args, "female_attr_path"):
        print("female_attr_path: ", args.female_attr_path)
    
    if hasattr(args, "male_attr_path"):
        print("male_attr_path: ", args.male_attr_path)
    
    if hasattr(args, "neo_attr_path"):
        print("neo_attr_path: ", args.neo_attr_path)
        
    print("templates_path: ", args.templates_path)
    print("target_path: ", args.target_path)
    print("output_path: ", args.output_path)
    
    print("=" * 100)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--male_attr_path",
                        help="path to male gendered words")
    
    parser.add_argument("--female_attr_path",
                        help="path to female gendered words")
    
    parser.add_argument("--neo_attr_path",
                        help="path to neo pronouns")
    
    parser.add_argument("--templates_path",
                        help="path to templates",
                        required=True)
    
    parser.add_argument("--target_path",
                        help="path to character/personality traits",
                        required=True)
    
    parser.add_argument("--output_path",
                        help="path to save output (gender-trait association scores)",
                        required=True)
    
    args = parser.parse_args()
    
    print_args(args)
    
    return args

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def create_main_gender_dict(gender_values, gender_option):
    '''
    Creates a dictionary mapping gendered words to a given gender label (male/female).
    
    Parameters:
    - gender_values: List of gendered words (e.g., ["she", "mother", "women"])
    - gender_option: Either "male" or "female"
    
    Returns:
    - A dictionary mapping each gendered word 
    (with unique suffixes for same gendered word mapped to multiple another gendered word)
    to the specified gender label.
    '''
    
    main_gender_dict = {}
    word_count = {}
    
    for g in gender_values:
        # keep track of multiple mapping
        if g in word_count:
            word_count[g] += 1
        else:
            word_count[g] = 1
        
        # assign _1 for single mapping (e.g., woman-man)
        # same gendered mapped to multiple another gender so we create unique gendered name for later mapping
        # (e.g., gal-dude, gal-guy) i.e., # gal_1, gal_2
        
        key = f"{g}_{word_count[g]}"
        
        main_gender_dict[key] = gender_option
        
    return main_gender_dict

def load_binary_gendered_words(args):
    """
    Loads male and female gendered words and returns a merged gender dictionary.
    """
    # Load female attributes
    with open(args.female_attr_path) as f:
        females = [line.strip() for line in f.readlines()]

    # Load male attributes
    with open(args.male_attr_path) as m:
        males = [line.strip() for line in m.readlines()]
    
    # Create gendered word dictionaries
    main_female_gender_dict = create_main_gender_dict(gender_values=females, gender_option="female")
    main_male_gender_dict = create_main_gender_dict(gender_values=males, gender_option="male")

    # merge male and female 
    main_gender_dict = merge_dicts(main_male_gender_dict, main_female_gender_dict)

    print("\nMain Gender Dictionary:", main_gender_dict, len(main_gender_dict), flush=True)

    return main_gender_dict

def load_neo_pronouns(args):
    """
    Loads neo-pronouns and returns a dictionary mapping each pronoun to the label 'neo'.
    """
    with open(args.neo_attr_path) as n:
        neo = [line.strip() for line in n.readlines()]
    
    neo_pronoun_dict = {}
    
    for pronoun in neo:
        neo_pronoun_dict[pronoun] = 'neo'
    
    print("\nNeo Pronoun Dictionary:", neo_pronoun_dict, len(neo_pronoun_dict), flush=True)
    
    return neo_pronoun_dict

def load_templates(args):
    '''
    Loads templates from a text file and returns a dictionary mapping each template 
    to a unique key (e.g., 'template_1', 'template_2', etc.)
    '''
    print("\nLoading templates...")
    
    with open(args.templates_path, 'r') as f:
        templates = [line.strip() for line in f]

    templates_dict = {f"template_{i+1}": template for i, template in enumerate(templates)}
    
    print("\nTemplates:", templates_dict, flush=True)

    return templates_dict

def load_targets(args):
    '''
    loads character trait/personality trait from a text file
    '''
    print("\nLoading targets...")

    with open(args.target_path, 'r') as f:
        targets = [line.strip() for line in f]
    
    print("Targets:", targets)
    
    return targets

def calculate_loss_and_perplexity(model, tokenizer, sentence):
    
    tokens = tokenizer(sentence, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens['input_ids'])
    
    loss = outputs.loss.item()
    perplexity = torch.exp(outputs.loss).item()

    return loss, perplexity
    
def main():
    
    args = parse_arguments()

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token)
    model.to(device)
    
    model.eval()

    templates_dict = load_templates(args)  
    
    # load gendered words (binary)
    if args.male_attr_path and args.female_attr_path:
        main_gender_dict = load_binary_gendered_words(args)
    
    # load gendered words (neo pronouns)
    elif args.neo_attr_path:
        main_gender_dict = load_neo_pronouns(args)
    
    targets = load_targets(args) # human traits
    
    # factor1: empathy, factor2: order, factor3: resourceful, factor4: serenity
    # factor1: extroversion, factor2: agreeableness, factor3: conscientiousness, factor4: emotional stability, factor5: openness    
    factor = os.path.splitext(os.path.basename(args.target_path))[0] # extract factor name: factor1, factor2, factor3, factor4
    
    ddf = []

    for template_id, template in templates_dict.items():
        
        template_actual = template
        
        for target in targets:
            
            print("template_id: {}, trait: {}".format(template_id, target))
            
            for gender_value, gender in main_gender_dict.items():

                template = template_actual

                if gender in ['male', 'female']:
                    gender_value_ = gender_value[0:-2] # extract gendered without suffix _1, _2 (she_1 --> she)
                elif gender == 'neo':
                    gender_value_ = gender_value
                
                pronoun = ""
                
                # replace [PRONOUN] with his/her in template for male/female gendered word
                if "is known for" in template or "personality is valued at" in template or "because of" in template: 
                    if gender=='male':
                        pronoun= 'his'
                    elif gender=='female':
                        pronoun = 'her'
                    elif gender =='neo':
                        if gender_value_ =='vi':
                            pronoun = 'vis'
                        elif gender_value_ =='xe':
                            pronoun = 'xyr'
                        elif gender_value_ =='ey':
                            pronoun = 'eir'
                        elif gender_value_ =='ze':
                            pronoun = 'zir'
                        elif gender_value_ =='ae':
                            pronoun = 'aer'
                
                if gender_value_=="he" or gender_value_=='she':
                    if "because of" in template:
                        gender_value_ = "him" if gender == 'male' else "her"
                    elif "personality is valued at" in template:
                        template = template.replace("'s","")
                        gender_value_ = "his" if gender == 'male' else "her"
                        
                if gender_value_ in ['vi', 'xe', 'ey', 'ze', 'ae']:
                    # use accusative neo-pronoun for the template "People admire [neo-pronoun] because of [possessive dependent] personality."
                    if "because of" in template: 
                        if gender_value_ =='vi':
                            gender_value_ = 'vir'
                        elif gender_value_ =='xe':
                            gender_value_ = 'xem'
                        elif gender_value_ =='ey':
                            gender_value_ = 'em'
                        elif gender_value_ =='ze':
                            gender_value_ = 'zir'
                        elif gender_value_ =='ae':
                            gender_value_ = 'aer'
                            
                # identify article before trait word
                if is_vowel(target):
                    template = template.replace('[ARTICLE]', 'an')
                else:
                    template = template.replace('[ARTICLE]', 'a')

                determiners = ['the', 'my', 'your', 'our', 'their']
                
                if gender_value_ in ['he', 'she', 'his', 'him', 'her','xe','ey','ze','vi', 'ae','vir','xem','em','zir','aer']:
                    original_sentence = template.replace('[DETERMINER]', '')
                    original_sentence = original_sentence.replace('[AAA]', gender_value_).replace('[TTT]', target)
                    original_sentence = original_sentence.replace('[PRONOUN]', pronoun)
                    
                    determiner_value = 'NA' # no determiners for "he/she/neo pronouns" attributes
                    is_selected_determiner = True # set True for "he/she/neo pronouns" attributes
                    sentence_loss, ppl_score = calculate_loss_and_perplexity(model, tokenizer, original_sentence)

                    ddf.append(
                        np.array([
                            # 'template_id', 'factor', 'personality', 'determiner',
                            template_id, factor, target, determiner_value, 
                            
                            # 'is_selected_determiner', 'gender_value',
                            is_selected_determiner, gender_value_,
                            
                            # 'gender_value_main', 'sentence_loss', 'gender',  
                            gender_value, sentence_loss, gender,  
                            
                            # 'ppl_score', 'original_sentence'
                            ppl_score, original_sentence 
                        ]))
                    
                else:
                    determiner_scores = {}

                    for determiner in determiners:
                        
                        original_sentence = template.replace('[DETERMINER]', determiner + " ")
                        original_sentence = original_sentence.replace('[AAA]', gender_value_).replace('[TTT]', target)
                        original_sentence = original_sentence.replace('[PRONOUN]', pronoun)

                        # sentence loss as a proxy for association score
                        sentence_loss, ppl_score = calculate_loss_and_perplexity(model, tokenizer, original_sentence)
                        
                        determiner_scores[determiner] = [ppl_score, sentence_loss, original_sentence]
                                                                    
                    # select determiner based on least perplexity score
                    determiner_value = min(determiner_scores, key=lambda k: determiner_scores[k][0])
                    
                    # set True for selected determiner, otherwise False
                    for k, v in determiner_scores.items():
                        is_selected_determiner = (k == determiner_value)

                        ddf.append(
                            np.array([
                                # 'template_id', 'factor', 'trait', 'determiner',
                                template_id, factor, target, k, 
                                
                                # 'is_selected_determiner', 'gender_value',
                                is_selected_determiner, gender_value_, 
                                
                                # 'gender_value_main', 'sentence_loss'(association_score), 'gender', 
                                gender_value, v[1], gender,  
                                
                                # 'ppl_score', 'original_sentence'
                                v[0], v[2] 
                            ]))
                        
    # store sentence_loss (proxy for 'association_score') stored under column 'association_score'
    df = pd.DataFrame(ddf, columns=['template_id', 'factor', 'trait', 'determiner',
                                    'is_selected_determiner', 'gender_value',
                                    'gender_value_main', 'association_score', 'gender', 
                                    'ppl_score', 'original_sentence'
                      ])
    
    df.to_csv(args.output_path, sep='\t', index=False)
    
main()