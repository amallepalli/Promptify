import csv

input_file = "400_cot.csv"
output_file = "1200_cot.csv"

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    # This quoting mode will automatically quote only when necessary
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

    header = next(reader)
    writer.writerow(header)

    for idx, row in enumerate(reader, start=1):
        row[0] = str(idx)
        role_playing_prompt = ','.join(row[2:])  # Combine all remaining columns
        output_row = [row[0], row[1], role_playing_prompt]  # No manual quotes!
        writer.writerow(output_row)
