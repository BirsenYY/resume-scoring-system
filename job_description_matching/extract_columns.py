import csv

# Define input and output file paths
input_csv = '../data/scores_testResumes.csv'
output_txt = '../data/output_testresumes.txt'

# Open the input CSV and output text file
with open(input_csv, newline='', encoding='utf-8') as csvfile, open(output_txt, 'w', encoding='utf-8') as txtfile:
    reader = csv.reader(csvfile)
    
    for row in reader:
        if len(row) >= 3:
            # Extract the first and third columns
            txtfile.write(f"{row[0]}\t{row[2]}\n")
