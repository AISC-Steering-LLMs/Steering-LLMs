# Open the input and output files
with open('../data/inputs/honesty_contrastive.csv', 'r') as infile, open('../data/inputs/honesty_contrastive_formatted_intermediate.csv', 'w', newline='') as outfile:
    # Initialize an empty list to store the parts of an entry
    entry_parts = []

    for line in infile:
        # Remove all quote marks from the line
        line = line.replace('"', '').strip()
        # If the line is not empty, add its content to entry_parts
        if line:
            entry_parts.append(line)
        # If entry_parts has two parts, join them with a space and write them to the output file
        if len(entry_parts) == 2:
            outfile.write('"' + ' '.join(entry_parts) + '",\n')
            entry_parts = []

    # Write the last entry to the output file if it hasn't been written yet
    if entry_parts:
        outfile.write('"' + ' '.join(entry_parts) + '",\n')



with open('../data/inputs/honesty_contrastive_formatted_intermediate.csv', 'r') as infile, open('../data/inputs/honesty_contrastive_formatted_final.csv', 'w', newline='') as outfile:
    
    # Write the header to the output file
    outfile.write('Prompt,Ethical_Area,Positive\n')
    
    # Initialize a line counter
    line_counter = 1

    for line in infile:
        # If the line number is odd, append ', "Bad", 0,' to the end of the line
        if line_counter % 2 != 0:
            line = line.rstrip('\n') + '"Bad",0\n'
        # If the line number is even, append ', "Good", 1,' to the end of the line
        else:
            line = line.rstrip('\n') + '"Good",1\n'
        # Write the line to the output file
        outfile.write(line)
        # Increment the line counter
        line_counter += 1