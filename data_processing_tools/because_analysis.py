import csv

# Path to your original CSV file
input_csv_path = 'data/inputs/good_bad_interactive.csv'

# Path to the new CSV file that will be created
output_csv_path = 'data/inputs/good_bad_interactive_no_because.csv'

# Open the original CSV file to read and a new CSV file to write
with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
     open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # Replace " because" with "." in the first column
        row[0] = row[0].replace(" because", ".")
        
        # Write the modified row to the new CSV file
        writer.writerow(row)

print("CSV has been processed and saved as a new file.")
