import sys
import os
import random
import numpy as np

# Global Variables
num_snippets = 0
mutation_rate = 0.0
from_ends = 0
output_file = ""
symbols = ["A", "B", "C", "D"]


def parse_args():
    global num_snippets, mutation_rate, from_ends, output_file
    num_snippets = int(sys.argv[1])
    mutation_rate = float(sys.argv[2])
    from_ends = int(sys.argv[3])
    output_file = sys.argv[4]


def print_help():
    print "python sticky_snippet_generator.py num_snippets mutation_rate from_ends output_file"


def generate_snippets():
    global symbols
    global mutation_rate
    global from_ends
    seed_string_list = []
    data = []
    # Generate sticky palindrome
    # First, we generate random string of length 20
    for x in range(20):
        seed_string_list.append(symbols[random.randint(0, 3)])
    # Create sticky second half
    sticky_half = []
    for item in seed_string_list:
        if item == 'A':
            sticky_half.append('C')
        elif item == 'B':
            sticky_half.append('D')
        elif item == 'C':
            sticky_half.append('A')
        elif item == 'D':
            sticky_half.append('B')
    # Reverse the sticky half and concatenate with seed list
    seed_string_list = seed_string_list + sticky_half[::-1]
    data.append(seed_string_list)
    # Generate other snippets
    for i in range(num_snippets - 1):
        new_string = []
        for x in range(from_ends):
            # Change characters with probability mutation_rate
            random_probability = random.random()
            if random_probability <= mutation_rate:
                random_symbol = symbols[random.randint(0, 3)]
                # Check whether the new symbol generated is same as the old one
                while random_symbol == data[i][x]:
                    random_symbol = symbols[random.randint(0,3)]
                new_string.append(random_symbol)
            else:
                new_string.append(data[i][x])
        # Generate rest characters randomly
        for m in range(from_ends,40):
            new_string.append(symbols[random.randint(0, 3)])
        data.append(new_string)
    return data


def write_to_file(data):
    global output_file
    output_string = ""
    # Convert the list into string
    for item in data:
        string = ''.join(item)
        output_string = output_string + string + '\n'
    # Write string to file
    with open(output_file, "w") as text_file:
        text_file.write(output_string)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Invalid arguments"
        print_help()
    else:
        parse_args()
        data = generate_snippets()
        write_to_file(data)



