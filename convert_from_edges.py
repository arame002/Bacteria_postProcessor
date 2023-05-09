import csv

# Define a list of column indices to remove
remove_cols = [0, 2, 3, 4, 7, 11, 12]

# Set the input and output filenames
tag = "01mMmal"
for index in range (1,8):
    input_file = f"{tag}{index}.csv"
    output_file = input_file.replace('.csv', '.txt')

    # Open the CSV file for reading
    with open(input_file, newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile)
        
        # Skip the first 4 rows
        for i in range(4):
            next(reader)
            
        # Open the output file for writing
        with open(output_file, 'w') as outfile:
            # Iterate over each row in the CSV file
            for row in reader:
                # Remove the specified columns
                row = [i for j, i in enumerate(row) if j not in remove_cols]
                
                # Write the modified row to the output file as tab-separated values
                outfile.write('\t'.join(row) + '\n')
