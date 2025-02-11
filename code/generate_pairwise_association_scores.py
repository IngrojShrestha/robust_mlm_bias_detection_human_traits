'''
This script formats the output for binary gender and neo-pronouns to generate files for pairwise gender bias analysis.  
It generates files for male_female, female_neo, and male_neo pairs.
'''
import pandas as pd
import argparse
import os

def process_files(binary_filepath, neo_filepath, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(binary_filepath):
        print(f"File {binary_filepath} not found.")
        return
    if not os.path.exists(neo_filepath):
        print(f"File {neo_filepath} not found.")
        return

    try:
        df_binary = pd.read_csv(binary_filepath, sep="\t")
        
        # extract association scores for neo-pronouns
        df_neo = pd.read_csv(neo_filepath, sep="\t")

        # extract association score for males
        df_male = df_binary[df_binary['gender'] == 'male']
        
        # extract association score for females
        df_female = df_binary[df_binary['gender'] == 'female']

        # pairwise association scores
        df_female_neo = pd.concat([df_female, df_neo], ignore_index=True)
        df_male_neo = pd.concat([df_male, df_neo], ignore_index=True)
        df_male_female = pd.concat([df_male, df_female], ignore_index=True)

        base_filename = os.path.basename(binary_filepath.replace("_binary", "")).replace(".tsv", "")
        
        df_male_female.to_csv(os.path.join(output_dir, f"{base_filename}_male_female.tsv"), sep="\t", index=False)
        df_female_neo.to_csv(os.path.join(output_dir, f"{base_filename}_female_neo.tsv"), sep="\t", index=False)
        df_male_neo.to_csv(os.path.join(output_dir, f"{base_filename}_male_neo.tsv"), sep="\t", index=False)

        print(f"Files saved in {output_dir}")

    except Exception as e:
        print(f"error processing files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process binary and neo association score output files and generate pairwise output files")
    
    parser.add_argument("binary_filepath", type=str, help="full path to the binary output file (TSV format)")
    parser.add_argument("neo_filepath", type=str, help="full path to the neo output file (TSV format)")
    parser.add_argument("output_dir", type=str, help="directory to save the processed output files")

    args = parser.parse_args()

    process_files(args.binary_filepath, args.neo_filepath, args.output_dir)