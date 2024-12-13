import os
import pandas as pd
import numpy as np
import scanorama
import argparse

def batch_correction(input_file):
    # Load the CSV file
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, index_col=0)

    # Filter by tissue types
    print("Filtering data by tissue types...")
    df_cl = df[df["tissue"] == "CL"]
    df_sb = df[df["tissue"] == "SB"]

    # Extract datasets and gene lists
    print("Preparing datasets for batch correction...")
    cl_ds = df_cl.iloc[:, :-6].values
    sb_ds = df_sb.iloc[:, :-6].values
    columns = df_cl.columns[:-6]

    datasets = [cl_ds, sb_ds]
    genes_list = [columns, columns]

    # Perform batch correction
    print("Applying batch correction using Scanorama...")
    corrected, genes = scanorama.correct(datasets, genes_list)

    # Save corrected data to CSV
    print("Saving corrected data...")
    corrected_df_cl = pd.DataFrame(corrected[0].toarray(), columns=genes, index=df_cl.index)
    corrected_df_sb = pd.DataFrame(corrected[1].toarray(), columns=genes, index=df_sb.index)

    corrected_df = pd.concat([corrected_df_sb, corrected_df_cl])
    corrected_df.to_csv("batch_corrected_data.csv")

    print("Batch correction completed. Output saved to 'batch_corrected_data.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch correction using Scanorama.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    batch_correction(args.input_file)
