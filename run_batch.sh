#!/bin/bash

set -e

# Input file and Python script
input_file="batch_input.txt"
script_1="bia_bmz_benchmark"
script_2="amalgamate_jsons"

# Loop through each line of the input file
while IFS= read -r line; do
    eval "$script_1" "$line"
done < "$input_file"

eval "$script_2"
