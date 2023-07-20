#!/bin/bash

# Get the name of the ipynb file as the first argument
ipynb_file=$1

# Loop through all the attachment:assets/* patterns in the ipynb file
grep -o "attachment:assets/[^)]*" $ipynb_file | while read -r attachment; do
	file_name=${attachment#attachment:}
	data_uri=$(grep -A2 "\"$file_name\"" $ipynb_file | grep "image/png" | cut -d'"' -f4)
	data_uri=$(echo "$data_uri" | sed 's/\\n//g')
	data_uri="data:image/png;base64,$data_uri"
	sed -i "s|$attachment|$data_uri|g" $ipynb_file
done

