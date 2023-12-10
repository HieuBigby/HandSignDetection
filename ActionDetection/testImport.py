import numpy as np

# replacements = {
#         "Toi Ten": "Toi ten la",
#         "Vui Gap": "Rat vui duoc gap ban"
#         # Thêm các cụm từ và cụm từ thay thế khác vào đây
#     }
#
# with open("structure.txt", 'w') as f:
#     for key, value in replacements.items():
#         f.write('%s: %s\n' % (key, value))

# actions = np.array(['None', 'A', 'B', 'C', 'D', 'E', 'I', 'H', 'U', 'Xin chao', 'Toi', 'Ten',
#                     'Cam on', 'Hen', 'Gap', 'Lai', 'Vui', 'Khoe', 'Xin loi', 'Tam biet'])
#
# with open("dictionary.txt", 'w') as f:
#     for index, action in enumerate(actions):
#         f.write('%s: %s\n' % (action, index))

# # Initialize an empty dictionary
# my_dict = {}
#
# # Open the file for reading
# with open('structure.txt', 'r', encoding='utf-8') as file:
#     # Read each line in the file
#     for line in file:
#         # Split each line into key and value based on the colon (':') character
#         key, value = map(str.strip, line.split(':', 1))
#
#         # Add the key-value pair to the dictionary
#         my_dict[key] = value
#
#     print(my_dict)

# Initialize an empty dictionary
my_list = []

# Open the file for reading
with open('dictionary.txt', 'r', encoding='utf-8') as file:
    # Read each line in the file
    for line in file:
        # Split each line into key and value based on the colon (':') character
        key, value = map(str.strip, line.split(':', 1))

        # Add the key-value pair to the dictionary
        my_list.append(key)

    print(np.array(my_list))