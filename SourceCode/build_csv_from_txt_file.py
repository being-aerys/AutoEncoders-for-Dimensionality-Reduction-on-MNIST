#-------------------------This thing literally took more than 6 hours. Can you believe it?


#--------------------------read from text file and get the data in the required format


import csv
import time
myfilename = "../data/rover_domain_datasets/data_4_10_20.txt"
f=open(myfilename,'r')
line = f.readline()
line_cnt = 1
states=[]
temp_line=[]
while line:
    line = line.replace('[', "")
    line = line.replace('\n', "")
    line = line.replace(']', "")
    line = line.split()

    for item in line:
        temp_line.append(item)

    if line_cnt%7 == 0:
        states.append(temp_line)
        temp_line=[]

    line = f.readline()
    line_cnt += 1

#---------------------------------Write the joint state spaces into a CSV

with open('csv_for_joint_states.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(states)

#----------------------------------Read the joint state spaces from the csv
with open("csv_for_joint_states.csv","r") as readfile:
    reader = csv.reader(readfile)
    # for index,data in enumerate(reader):
    #     print(data)
    #     time.sleep(5)

