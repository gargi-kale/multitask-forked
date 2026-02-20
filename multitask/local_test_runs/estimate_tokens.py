#!/usr/bin/env python3
"""
Estimate the number of tokens in no_prop_overlap_*.csv files using tiktoken.
Focuses on the 'text' column in each CSV.
"""

import csv
import glob
from pathlib import Path
import tiktoken

def count_tokens_in_text(text: str, encoding) -> int:
    """Count the number of tokens in a given text."""
    return len(encoding.encode(text))

def estimate_tokens_in_csvs():
    """Process all no_prop_overlap_*.csv files and estimate token counts."""
    # Initialize tiktoken encoder (using cl100k_base for OpenAI models)
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Get the data directory path
    data_dir = Path(__file__).parent.parent / "data"
    
    # Find all no_prop_overlap_*.csv files
    csv_files = sorted(glob.glob(str(data_dir / "no_prop_overlap_*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):\n")
    
    total_tokens = 0
    
    for csv_path in csv_files:
        file_name = Path(csv_path).name
        token_count = 0
        row_count = 0
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'text' in row:
                        text = row['text']
                        token_count += count_tokens_in_text(text, encoding)
                        row_count += 1
            
            total_tokens += token_count
            print(f"{file_name}:")
            print(f"  Rows: {row_count}")
            print(f"  Total tokens: {token_count:,}")
            print(f"  Avg tokens per row: {token_count / row_count:.1f}" if row_count > 0 else "")
            print()
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}\n")
    
    print(f"Summary:")
    print(f"  Total tokens across all files: {total_tokens:,}")

if __name__ == "__main__":
    estimate_tokens_in_csvs()
