import argparse
import json

def extract_jsonl_entries(input_path, output_path, indices):
    """
    Reads a JSONL file and writes specified entries to a new file.

    Args:
        input_path (str): The path to the source .jsonl file.
        output_path (str): The path to the destination .jsonl file.
        indices (list[int]): A list of 0-indexed line numbers to extract.
    """
    # Convert the list of indices to a set for efficient O(1) lookups.
    indices_to_extract = set(indices)
    
    # Keep track of how many matching entries we've found and written.
    entries_written = 0

    print(f"Attempting to extract {len(indices_to_extract)} entries from '{input_path}'...")

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                # If the current line number is one we're looking for...
                if i in indices_to_extract:
                    # ...write it to the output file.
                    outfile.write(line)
                    entries_written += 1
                    # A small optimization: if we've found all requested indices, we can stop reading.
                    if entries_written == len(indices_to_extract):
                        print("All specified entries have been found and written.")
                        break

        print(f"\nSuccess! Wrote {entries_written} entries to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up the command-line argument parser.
    parser = argparse.ArgumentParser(
        description="Extract specific entries from a JSONL file based on their 0-indexed line number.",
        epilog="Example: python extract_jsonl.py data.jsonl output.jsonl --indices 1 3 4"
    )
    
    # Required positional arguments for file paths
    parser.add_argument("input_file", help="The path to the input .jsonl file.")
    parser.add_argument("output_file", help="The path for the output .jsonl file.")
    
    # Required named argument for indices
    parser.add_argument(
        "--indices",
        required=True,
        nargs='+',  # This allows for one or more space-separated values
        type=int,
        help="A space-separated list of 0-indexed entry numbers to extract."
    )
    
    args = parser.parse_args()
    
    # Run the main function with the parsed arguments.
    extract_jsonl_entries(args.input_file, args.output_file, args.indices)
