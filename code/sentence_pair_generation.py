'''
Section: "Sentence generation from templates": 
When selections differ across genders, 
e.g., my father and your mother, we add the alternatives your father and my mother for balance.
'''
import pandas as pd
import numpy as np
import os
import functools

import argparse

print = functools.partial(print, flush=True)

def print_args(args):
    print("=" * 100)
    print("male_attr_path: ", args.male_attr_path)
    print("female_attr_path: ", args.female_attr_path)
    print("in_file_path: ", args.in_file_path)
    print("output_path: ", args.output_path)
    print("=" * 100)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--male_attr_path",
                        help="path to male gendered words",
                        required=True)
    parser.add_argument("--female_attr_path",
                        help="path to female gendered words",
                        required=True)
    parser.add_argument("--in_file_path",
                            help="e.g., factor1_binary.tsv",
                            required=True)
    parser.add_argument("--output_path",
                        help="e.g., factor1_paired_true_cases_binary.tsv",
                        required=True)

    args = parser.parse_args()
    return args

def get_all_indices(alist, item_to_find):
    indices = []
    for idx, value in enumerate(alist):
        if value == item_to_find:
            indices.append(idx)
    return indices

def fill_non_matching_determiner(df, df_m_id, df_f_id):
    # male
    df_m = df.iloc[[df_m_id]]
    t1 = df_m['template_id'].values[0]
    f1 = df_m['factor'].values[0]
    p1 = df_m['trait'].values[0]
    gmain1 = df_m['gender_value_main'].values[0]
    d1 = df_m['determiner'].values[0]
    isd1 = bool(df_m['is_selected_determiner'].values[0])

    df_f = df.iloc[[df_f_id]]
    t2 = df_f['template_id'].values[0]
    f2 = df_f['factor'].values[0]
    p2 = df_f['trait'].values[0]
    gmain2 = df_f['gender_value_main'].values[0]
    d2 = df_f['determiner'].values[0]
    isd2 = bool(df_f['is_selected_determiner'].values[0])

    if isd1 and isd2:
        pass

    else:
        if isd1:
            df2.at[df_f_id, 'is_selected_determiner'] = True

            # extract true determiner from female
            df_f_temp = df[(df['template_id'] == t2) &
                           (df['factor'] == f2) &
                           (df['trait'] == p2) &
                           (df['gender_value_main'] == gmain2)]

            # print("df_f_temp: ",df_f_temp)

            df_f_temp = df_f_temp[df_f_temp['is_selected_determiner'] == True]
            # print()
            # print("df_f_temp: ",df_f_temp)

            df_f_temp_determiner = df_f_temp['determiner'].values[0]
            # print()
            # print("df_f_temp_determiner: ",df_f_temp_determiner)

            df_m_temp = df[(df['template_id'] == t1) &
                           (df['factor'] == f1) &
                           (df['trait'] == p1) &
                           (df['determiner'] == df_f_temp_determiner) &
                           (df['gender_value_main'] == gmain1)]

            # print()
            # print("df_m_temp: ", df_m_temp)
            df2.at[df_m_temp.index[0], 'is_selected_determiner'] = True

        else:
            df2.at[df_m_id, 'is_selected_determiner'] = True

            # extract true determiner from female
            df_m_temp = df[(df['template_id'] == t1) &
                           (df['factor'] == f1) &
                           (df['trait'] == p1) &
                           (df['gender_value_main'] == gmain1)]

            df_m_temp = df_m_temp[df_m_temp['is_selected_determiner'] == True]

            df_m_temp_determiner = df_m_temp['determiner'].values[0]

            df_f_temp = df[(df['template_id'] == t2) &
                           (df['factor'] == f2) &
                           (df['trait'] == p2) &
                           (df['determiner'] == df_m_temp_determiner) &
                           (df['gender_value_main'] == gmain2)]
            df2.at[df_f_temp.index[0], 'is_selected_determiner'] = True

def main_fill_determiners(male_attr_path,female_attr_path,in_filename):
    #################### Load output genderated on runnining MLM ####################
    global df2

    df = pd.read_csv(in_filename, sep='\t')
    df2 = pd.read_csv(in_filename, sep='\t')
    
    # female gendered words
    with open(female_attr_path) as f:
        females = [line.strip() for line in f]
        
    # male gendered words
    with open(male_attr_path) as m:
        males = [line.strip() for line in m]

    counterpart_dict = {}

    # for multiple mapping between gendered words (e.g., guy and dude mapped to gal)
    # extract the corresponding pairs (guy_1, gal_1, dude_1, gal_2)
    for idx, m in enumerate(males):
        if m + "_1" not in counterpart_dict:
            counterpart_dict[m + "_1"] = females[idx]
        elif m + "_2" not in counterpart_dict:
            counterpart_dict[m + "_2"] = females[idx]
        elif m + "_3" not in counterpart_dict:
            counterpart_dict[m + "_3"] = females[idx]
        else:
            counterpart_dict[m + "_4"] = females[idx]

    ########################################
    count = 0
    for index, row in df.iterrows():

        df_m_id = index

        template_id = row['template_id']
        factor = row['factor']
        trait = row['trait']
        is_selected = bool(row['is_selected_determiner'])
        determiner = row['determiner']
        m_gender_value_main = row['gender_value_main']
        m_gender_value = row['gender_value']
        
        if m_gender_value =="ballet▁dancer":
            m_gender_value = "ballet_dancer" # revert back to underscore (change "▁" use in albert to "_" underscore)

        gender = row['gender']

        if gender == 'female':
            continue

        if is_selected and (m_gender_value not in ['he','his','him']):
            counterparts = counterpart_dict[m_gender_value_main]

            print(f"gender_value_main: {m_gender_value_main}, selected_determiner: {determiner}")
            print(f"femlae_counterparts: ", counterparts)

            female_filtered_dict = df2[(df2['template_id'] == template_id) &
                                       (df2['factor'] == factor) &
                                       (df2['trait'] == trait) &
                                       (df2['determiner'] == determiner) &
                                       (df2['gender_value'] == counterparts)]
            print("female_filtered_dict.index: ", female_filtered_dict.index)
            if len(female_filtered_dict.index) > 1:
                count += 1

            f_gender_value_main = female_filtered_dict['gender_value_main']

            m_indices = get_all_indices(males, m_gender_value)
            if m_gender_value_main[-1:] == '1':
                m_idx = m_indices[0]
            elif m_gender_value_main[-1:] == '2':
                m_idx = m_indices[1]
            elif m_gender_value_main[-1:] == '3':
                m_idx = m_indices[2]
            elif m_gender_value_main[-1:] == '4':
                m_idx = m_indices[3]

            female_counterpart = females[m_idx]

            print("m_indices: ", m_indices)
            # print("female_counterpart: ",female_counterpart)

            # find all possible indices fo female_counterparts in female attribute values
            f_indices = get_all_indices(females, female_counterpart)

            # intersection of m_indices and f_indices
            common_id = list(set([m_idx]).intersection(set(f_indices)))

            df_f_id = female_filtered_dict.index[f_indices.index(common_id[0])]

            print(f"m_idx: {m_idx}, f_indices: {f_indices}")
            print("common_id: ", common_id)
            print("actual female df_index in df: ", df_f_id)
            print("actual male df_index in df: ", df_m_id)

            # ==================================================================
            # Extract df_m and df_f using actual id
            fill_non_matching_determiner(df, df_m_id, df_f_id)

            print("=" * 50)

    # print(count)
    return df2

def main():
    
    args = parse_arguments()
    
    print_args(args)
    
    paired_df = main_fill_determiners(male_attr_path=args.male_attr_path,
                          female_attr_path=  args.female_attr_path,
                          in_filename= args.in_file_path)
    
    # generate factor1_paired_true_cases.tsv
    filtered_df = paired_df[paired_df['is_selected_determiner']]
    filtered_df.to_csv(args.output_path, sep='\t', index=False)
    
main()