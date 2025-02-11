import pandas as pd
import argparse

def combine_positive_negative_traits_results(pos_trait_path, neg_trait_path, output_path):
    """
    Combines results for positive and negative traits
    """
    df_positive = pd.read_csv(pos_trait_path, sep="\t")
    df_negative = pd.read_csv(neg_trait_path, sep="\t")
    
    df_positive['trait_direction'] = "positive"
    df_negative["trait_direction"] = "negative"
    
    df_negative['association_score'] = df_negative['association_score'] * -1
    
    df_combined = pd.concat([df_positive, df_negative], ignore_index=True)
    df_combined.to_csv(output_path, sep="\t", index=False)
    
    print(f'File saved to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine positive and negative personality trait output files")
    
    parser.add_argument("--pos_trait_path", type=str, required=True, help="path to the positive trait file (TSV format).")
    parser.add_argument("--neg_trait_path", type=str, required=True, help="path to the negative trait file (TSV format).")
    parser.add_argument("--output_path", type=str, required=True, help="path to save the combined output file (TSV format).")

    args = parser.parse_args()
    
    combine_positive_negative_traits_results(args.pos_trait_path, args.neg_trait_path, args.output_path)
