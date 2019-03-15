# Read Text File
# Created by Ashwin Vinoo
# Date 3/9/2019

# ------ Hyper Parameters -----
file_to_read = 'data_4_10_20.txt'


# This function reads in the input file and converts it
def file_reader(file_location):

    # This list can be used to identify the characters that are a part of the numbers
    number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '-']

    # We open the file in read mode
    file = open(file_location, 'r')
    # This string holds the entire file text
    accumulated_text = ''
    # We iterate through the file line by line
    for line in file:
        accumulated_text += line
    # We close the connection to the file
    file.close()

    # This variable holds the state values for the entire file as a list
    game_list = []
    # This variable holds the state values for a round with all the agents
    round_list = []
    # This variable holds the state value of a single agent as a list
    agent_list = []

    # This flag measures the access to the outer list via '[' and ']'
    layer_1_flag = False
    # This flag measures the access to the inner list via '[' and ']'
    layer_2_flag = False
    # This flag helps us identify a number
    number_flag = False
    # The current number we are building by appending individual characters
    current_number = ''

    # We iterate through all the characters in the list
    for character in accumulated_text:

        # We check if the character is in the number list (we entered the text of a number)
        if character in number_list:
            # We mark that we entered a number
            number_flag = True
            # We append the character to build up the current number
            current_number += character
        # We check if the character is not in the number list (we have left the text of a number)
        elif character not in number_list and number_flag:
            # We mark that we left a number
            number_flag = False
            # We add the floating point version of the string to the agent list
            agent_list.append(float(current_number))
            # The current number string is initialized back to ''
            current_number = ''
        # We check if we have entered into the outer list
        elif character == '[' and not layer_1_flag:
            # We mark that we are in the outer list
            layer_1_flag = True
            # We empty the round list
            round_list = []
        # We check if we have entered into the inner list
        elif character == '[' and layer_1_flag:
            # We mark that we are in the inner list
            layer_2_flag = True
            # We empty the agent list
            agent_list = []
        # We check if we have exited the inner list
        elif character == ']' and layer_2_flag:
            # We mark that we have exited the inner list
            layer_2_flag = False
            # We add the agent list to the round list
            round_list.append(agent_list)
            if number_flag:
                # We mark that we left a number
                number_flag = False
                # We add the floating point version of the string to the agent list
                agent_list.append(float(current_number))
                # The current number string is initialized back to ''
                current_number = ''
        # We check if we have exited the outer list
        elif character == ']' and layer_1_flag:
            # We mark that we have exited the outer list
            layer_1_flag = False
            # We add the round list to the game list
            game_list.append(round_list)

    # We return the game list
    return game_list

# If this file is the main one called for execution
if __name__ == "__main__":
    # We attempt to read the file
    game_list = file_reader(file_to_read)

