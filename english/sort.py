
with open('data.csv', mode='r', newline='') as file:
    content = file.read().strip().split("\n")

paired_lines = []
current_term = None
current_def = None

for line in content:
    line = line.strip()
    if line:
        if current_term is None and not ":" in line:
            current_term = line
        elif "Definition" in line:
            current_def = line
        elif "Example" in line:
            paired_lines.append(f"{current_term}: {current_def.split(":")[1]} \\n e.g {line.split(":")[1]}")
            current_term = None
            current_def = None

with open('paired_output3.txt', mode='w') as output_file:
    for paired_line in paired_lines:
        output_file.write(paired_line + '\n')
