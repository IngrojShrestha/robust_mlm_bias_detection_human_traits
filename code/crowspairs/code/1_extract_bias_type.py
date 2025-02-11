import pandas as pd
from xlwings import view
import sys
import os

def extract_save_data_bias_type(bias_type, input_file, output_file):
    
    folder_path = os.path.dirname(output_file) + "/"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    df_main = pd.read_csv(input_file, sep=",")

    df = df_main[df_main['bias_type']==bias_type]

    # Calculate the sentence length by counting words in each sentence
    df['len_sent_more'] = df['sent_more'].str.split().str.len().astype(int)
    df['len_sent_less'] = df['sent_less'].str.split().str.len().astype(int)

    # Analyze the distribution to determine bin edges based on quantiles
    quantiles = df['len_sent_more'].quantile([0.33, 0.67]).round().astype(int).tolist()

    # Get the minimum and maximum values from the data and round them
    min_length = int(df['len_sent_more'].min())
    max_length = int(df['len_sent_more'].max())

    # Ensure the bins cover the entire range of data
    bins = [min_length - 1] + quantiles + [max_length + 1]  # Adjust to ensure inclusion of max length

    # Define labels for bins. You need three intervals, thus three labels
    labels = ['Short', 'Medium', 'Long']


    ranges_df = pd.DataFrame({
        'Label': labels,
        'Range (Lower Bound)': bins[:-1],
        'Range (Upper Bound)': bins[1:]
    })

    print(ranges_df)
    # for labels, bins in zip(labels, bins):
    #     print(f'{labels}: {bins}')

    # Assign categories based on len_sent_more column
    df['sent_length_cat'] = pd.cut(df['len_sent_more'], bins=bins, labels=labels, right=True)

    df.to_csv(output_file, sep='\t', index=False)


bias_type = sys.argv[1].strip() # ['age', 'disability', 'gender', 'nationality', 'physical-appearance', 'race-color', 'religion', 'sexual-orientation', 'socioeconomic']
    
if __name__ == "__main__":
    print("Extracting bias type data...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias_type", type=str, help="bias type (e.g., race, gender)")
    parser.add_argument("--input_file", type=str, help="Path to input file")
    parser.add_argument("--output_file", type=str, help="Path to output files storing specific bias types data")
    args = parser.parse_args()
    extract_save_data_bias_type(args.bias_type, args.input_file, args.output_file)
