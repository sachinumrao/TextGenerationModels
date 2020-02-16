import contractions

with open('../../../Data/LM/friends_script.txt', 'r') as data_file:
    data = data_file.read()

# Replace common unwanted occurrences"
replace_str1 = "[Time Lapse "
target_str1 = "["

replace_str2 = "Closing Credits"
target_str2 = ""

replace_str3 = "end"
target_str3 = ""

replace_str4 = "[Time Lapse]"
target_str4 = ""

data.replace(replace_str1, target_str1)
data.replace(replace_str2, target_str2)
data.replace(replace_str3, target_str3)
data.replace(replace_str4, target_str4)

# Add additional cleaning steps
data = contractions.fix(data)

# Save editted text data
edit_data_file = open("../../../Data/LM/friends_script_editted.txt", "w")
edit_data_file.write(data)
edit_data_file.close()