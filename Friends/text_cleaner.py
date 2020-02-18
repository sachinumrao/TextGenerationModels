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

replace_str5 = "Commercial Break"
target_str5 = ""

data = data.replace(replace_str1, target_str1)
data = data.replace(replace_str2, target_str2)
data = data.replace(replace_str3, target_str3)
data = data.replace(replace_str4, target_str4)
data = data.replace(replace_str5, target_str5)

# Save editted text data
with open("../../..//Data/LM/friends_script_edit.txt", "w") as edit_data_file:
    edit_data_file.write(data)
