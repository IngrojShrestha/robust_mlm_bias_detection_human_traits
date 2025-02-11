import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime
import functools
import torch 
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM, AlbertForMaskedLM, AlbertTokenizer

print = functools.partial(print, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        help="bert-base-uncased, bert-large-uncased, roberta-base, roberta-large distilbert-base-uncased, albert-base-v2, albert-large-v2",
                        required=True)
     
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

def print_args(args):
    
    print("=" * 100)
    
    print("model: ", args.model)
    
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

def get_last(template):
    last = False
    if template.find('[AAA]') > template.find('[TTT]'):
        last = True
    return last

def get_num_subwords(tokenizer, word):
    return len(tokenizer.tokenize(word))

def find_mask_token(tokenizer, sentence, attribute_num, MSK, last=False, target_num=0):
    tokens = tokenizer.encode(sentence)
    # skip = 0
    for i, tk in enumerate(tokens):
        if tk == MSK:
            if last == False:
                return list(range(i, i + attribute_num))
            else:
                pass
                # if skip != (target_num - 1):
                #     skip += 1
                #     continue
                # else:
                #     last = False
                #     skip += 1

def get_perplexity_score(model, tokenizer, sentence):

    tensor_input = tokenizer.encode(sentence, return_tensors='pt').to(device)
    
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1).to(device)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2].to(device)
    
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id).to(device)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100).to(device)
    
    loss = model(masked_input, labels=labels)[0].to(device)
    perplexity = np.exp(loss.item())
    
    return perplexity

def get_association_score(model, tokenizer, gen, sentence, prior_sentence,
                         attribute_num, target_num, last, is_gender_value_start_of_sentence):
    
    vocab = tokenizer.get_vocab()
    softmax = torch.nn.Softmax()

    input_ids = tokenizer(sentence, return_tensors='pt').to(device)

    #  same as target_prob = model(**input_ids).logits
    # provides (batch_size, num_tokens, embedding_dim)
    target_prob = model(**input_ids)[0].to(device)

    prior_input_ids = tokenizer(prior_sentence, return_tensors='pt').to(device)

    #  same as prior_prob = model(**prior_input_ids).logits
    # provides (batch_size, num_tokens, embedding_dim)
    prior_prob = model(**prior_input_ids)[0].to(device)

    # attribute_num = number of masked tokens for gendered word
    masked_tokens = find_mask_token(tokenizer, sentence, attribute_num, tokenizer.mask_token_id)

    # target_num = number of masked tokens for target (trait word)
    masked_tokens_prior = find_mask_token(tokenizer, prior_sentence, attribute_num, tokenizer.mask_token_id, last, target_num)
    
    logits = []
    prior_logits = []
    for mask in masked_tokens:
        # here we take target_prob[0] to extract likelihoods for a sentence (num_tokens, embedding_dim)
        logits.append(softmax(target_prob[0][mask]).detach())

    for mask in masked_tokens_prior:
        # here we take prior_prob[0] to extract likelihoods for a sentence (num_tokens, embedding_dim)
        prior_logits.append(softmax(prior_prob[0][mask]).detach())

    gen_logit = 1.0
    gen_prior_logit = 1.0

    # Check if the tokenizer is specifically for RoBERTa
    if isinstance(tokenizer, RobertaTokenizer):
        # RoBERTa: if a word is at the start of the sentence then it does not consider space while tokenization
        if is_gender_value_start_of_sentence:
            gender_value = gen
        else:
            # RoBERTa: if a word is not start of the sentence then it considers space in tokenization 
            gender_value = " " + gen
    else:
        # For BERT, DistilBERT, ALBERT no special handling needed
        gender_value = gen
        
    for token in tokenizer.tokenize(gender_value): # white space is added as roberta considers white space while tokenizing.
        for logit in logits:
            gen_logit *= float(logit[vocab[token]].item())
        for prior_logit in prior_logits:
            gen_prior_logit *= float(prior_logit[vocab[token]].item())

    return np.log(float(gen_logit / gen_prior_logit))
    
def main():
    
    args = parse_arguments()

    # load tokenizer and model
    if args.model.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(args.model)
        model = BertForMaskedLM.from_pretrained(args.model)
        
    # load pre-trained or debiased-roberta
    elif args.model.startswith('roberta') or 'debiased-roberta' in args.model:
        
        # for loading debiased roberta provide path to finetunedRL
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
        model = RobertaForMaskedLM.from_pretrained(args.model)

        # tokenizer = RobertaTokenizer.from_pretrained("finetunedRL")
        # model = RobertaForMaskedLM.from_pretrained("finetunedRL")
        
    elif args.model.startswith('distilbert'):
        tokenizer = DistilBertTokenizer.from_pretrained(args.model)
        model = DistilBertForMaskedLM.from_pretrained(args.model)
        
    elif args.model.startswith('albert'):
        tokenizer = AlbertTokenizer.from_pretrained(args.model)
        model = AlbertForMaskedLM.from_pretrained(args.model)
    else:
        raise NotImplementedError("Not implemented")

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
        
        last = get_last(template)
        
        for target in targets:
            
            target_num = get_num_subwords(tokenizer, target)
            
            print("template_id: {}, trait: {}".format(template_id, target))
            
            for gender_value, gender in main_gender_dict.items():

                template = template_actual

                if gender in ['male', 'female']:
                    gender_value_ = gender_value[0:-2] # remove suffix _1, _2, _3 added for corresponding mapping between pair of gendered word
                elif gender == 'neo':
                    gender_value_ = gender_value
                
                # albert does not recognize "_"
                if isinstance(tokenizer, AlbertTokenizer) and gender_value_ == "ballet_dancer":
                    gender_value_ = "ballet▁dancer" # special underscore for albert
                    
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
                
                # mask pronoun
                pronoun_masked = " ".join([tokenizer.mask_token for _ in range(get_num_subwords(tokenizer, pronoun))])
                
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

                attribute_num = get_num_subwords(tokenizer, gender_value_)

                attribute_mask = ' '.join([tokenizer.mask_token for _ in range(attribute_num)])  # number of masked tokens for gendered words
                target_mask = ' '.join([tokenizer.mask_token for _ in range(target_num)])  # number of masked tokens for trait word
                
                sentence = template.replace('[AAA]', attribute_mask).replace('[TTT]', target)
                prior_sentence = template.replace('[AAA]',attribute_mask).replace('[TTT]', target_mask)
                
                determiners = ['the', 'my', 'your', 'our', 'their']
                
                if gender_value_ in ['he', 'she', 'his', 'him', 'her','xe','ey','ze','vi', 'ae','vir','xem','em','zir','aer']:
                    original_sentence = template.replace('[DETERMINER]', '')
                    original_sentence = original_sentence.replace('[AAA]', gender_value_).replace('[TTT]', target)
                    original_sentence = original_sentence.replace('[PRONOUN]', pronoun)
                    
                    # need to adjust vocab map based on whether gendered word occur at the start oof the sentence or not
                    if original_sentence.startswith(gender_value_):
                        is_gender_value_start_of_sentence = True
                    else:
                        is_gender_value_start_of_sentence = False
                        
                    sentence_ = sentence.replace('[DETERMINER]', '').replace('[PRONOUN]', pronoun_masked)
                    prior_sentence_ = prior_sentence.replace('[DETERMINER]', '').replace('[PRONOUN]', pronoun_masked)
                        
                    determiner_value = 'NA' # no determiners for "he/she/neo pronouns" attributes
                    is_selected_determiner = True # set True for "he/she/neo pronouns" attributes
                    
                    # get perplexity score
                    ppl_score = get_perplexity_score(model, tokenizer, original_sentence)

                    association_score = get_association_score(
                        model, tokenizer, gender_value_, sentence_,
                        prior_sentence_, attribute_num, target_num, last, is_gender_value_start_of_sentence)

                    ddf.append(
                        np.array([
                            # 'template_id', 'factor', 'trait', 'determiner',
                            template_id, factor, target, determiner_value,
                            
                            # 'is_selected_determiner', 'gender_value' (without suffix 1, 2, 3, 4),
                            is_selected_determiner, gender_value_,
                            
                            # 'gender_value_main', 'association_score', 'gender', 'model',
                            gender_value, association_score, gender, args.model,
                            
                            # 'ppl_score', 'original_sentence', 'gender_masked', 'prior_sent'
                            ppl_score, original_sentence, sentence_, prior_sentence_
                        ]))
                    
                else:
                    determiner_scores = {}

                    for determiner in determiners:
                        original_sentence = template.replace('[DETERMINER]', determiner + " ")
                        original_sentence = original_sentence.replace('[AAA]', gender_value_).replace('[TTT]', target)
                        original_sentence = original_sentence.replace('[PRONOUN]', pronoun)      
                        
                        determiner_scores[determiner] = [get_perplexity_score(model, tokenizer, original_sentence)]
                        
                        sentence_ = sentence.replace('[DETERMINER]', determiner + " ").replace('[PRONOUN]', pronoun_masked)
                        prior_sentence_ = prior_sentence.replace('[DETERMINER]', determiner + " ").replace('[PRONOUN]', pronoun_masked)
                            
                        # is_gender_value_start_of_sentence = False as there is always determiner before gendered word. Hence we have space before gendered word
                        association_score = get_association_score(model, tokenizer, gender_value_, sentence_,
                                                                  prior_sentence_, attribute_num, target_num, last, False)
                        
                        determiner_scores[determiner].extend([association_score, original_sentence, sentence_, prior_sentence_])
                                                                    
                    # select determiner based on least perplexity score
                    determiner_value = min(determiner_scores, key=lambda k: determiner_scores[k][0])
                    
                    # set True for selected determiner, otherwise False
                    for k, v in determiner_scores.items():
                        is_selected_determiner = (k == determiner_value)

    
                        ddf.append(
                            np.array([
                                # 'template_id', 'factor', 'trait', 'determiner'
                                template_id, factor, target, k,
                                
                                # 'is_selected_determiner', 'gender_value',
                                is_selected_determiner, gender_value_,
                                
                                # 'gender_value_main', 'association_score', 'gender', 'model',
                                gender_value, v[1], gender, args.model,
                                
                                # 'ppl_score', 'original_sentence', 'gender_masked', 'prior_sent'
                                v[0], v[2], v[3], v[4]
                            ]))
                                            
    df = pd.DataFrame(ddf, columns=['template_id', 'factor', 'trait', 'determiner',
                          'is_selected_determiner', 'gender_value',
                          'gender_value_main', 'association_score', 'gender', 'model',
                          'ppl_score', 'original_sentence', 'gender_masked', 'prior_sent'
                      ])
    
    df.to_csv(args.output_path, sep='\t', index=False)
    
main()