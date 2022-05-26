import pandas as pd
import xlsxwriter
import datetime
import numpy
import math
import random
pd.options.mode.chained_assignment = None  # default='warn'
workbook = xlsxwriter.Workbook('new.xlsx')
worksheet = workbook.add_worksheet()
'''
#For create fake RR file:
repo = xlsxwriter.Workbook('resp demo.xlsx')
repobook = repo.add_worksheet()
repobook.write('A1', 'h-num_demo')
repobook.write('B1', 'parameter')
repobook.write('C1', 'time')
repobook.write('D1', 'value')
'''
worksheet.write('A1', 'h-num_demo')
worksheet.write('B1', 'date')
worksheet.write('C1', 'counter')
worksheet.write('D1', 'HR')
worksheet.write('E1', 'MAP')
worksheet.write('F1', 'BSA')
worksheet.write('G1', 'RR')
worksheet.write('H1', 'CVRI')
worksheet.write('I1', 'EVENT')
drugs = [4704, 5433]
row_excel = 0


def put_id(total_row, id):
    i = row_excel
    j = 0
    BSA = calculate_BSA(id)
    #RR = get_RR_value(id)
    while i < row_excel + total_row:
        worksheet.write(i + 1, 0, id)  # patient number.
        #repobook.write(i + 1, 0, id) # add to RR file.
        #repobook.write(i + 1, 1, '4612') #add the number of drug.
        worksheet.write(i + 1, 2, j)  # counter.
        worksheet.write(i + 1, 5, BSA)  # BSA.
        #if id == repo_file['h-num_demo'][i]:

        #worksheet.write(i + 1, 6, get_RR_value(i)) # RR
         #   worksheet.write(i + 1, 6, repo_file['value'][i])
        i += 1
        j += 1


def put_value(index_of_drug_we_need, row_num_first, row_num_last):
    min_time = max(all_the_last_time)
    for i in index_of_drug_we_need:
        for j in i:
            min_time = min(min_time, all_the_first_time[j])
    # print(min_time)
    colomn = 3
    time_runner = 2
    keep_min = min_time
    for i in index_of_drug_we_need:
        for j in i:
            row_start = row_num_first[j] - 2
            row_finish = row_num_last[j] - 2
            keep_row = row_excel
            prev = 0
            while row_start < row_finish:
                time = vital_file['time'][row_start]
                while time + datetime.timedelta(hours=1) > min_time:
                    min_time = min_time + datetime.timedelta(hours=1)
                    keep_row += 1

                # worksheet.write(keep_row, colomn, prev)
                vital = vital_file['para_code'][row_start]
                value = vital_file['value_demo'][row_start]
                if validation(vital, value, keep_row, colomn):
                    worksheet.write(keep_row, colomn, value)
                    prev = value

                worksheet.write(keep_row, colomn - time_runner, str(time))  # Times into the Excel file.
                row_start += 1
            min_time = keep_min
        colomn += 1
        time_runner += 1


'''
Function to add '1' to the Event col and put red color into the cell.
'''


def put1andColor(row):
    format = workbook.add_format()
    format.set_pattern(1)
    format.set_bg_color('red')
    worksheet.write(row, 8, 1, format)


def validation(vital_number, value, row, colomn):
    if vital_number == drugs[0]:  # HR check 40-180
        if 40 <= int(value) <= 180:
            return True
    if vital_number == drugs[1]:  # MAP check 50-180
        if 50 <= int(value) <= 180:
            if value < 60:
                put1andColor(row)
            else:
                worksheet.write(row,8, 0)
            return True
    # if vital_number == drugs[2]:  # There is no check for now.
    #   return True
    return False

def calculate_BSA(patient_id):
    # excel_file = 'weight-height demo.xlsx'
    param = [6393, 6395]
    # WeightHeight_file = pd.read_excel(excel_file)
    weight = 0
    height = 0
    for i in range(len(WeightHeight_file['h_num_demo'])):
        if patient_id == WeightHeight_file['h_num_demo'][i]:
            if param[0] == WeightHeight_file['ParameterID'][i]:
                weight = WeightHeight_file['Value'][i]
            else:
                height = WeightHeight_file['Value'][i]
                break
    BSA = 0
    height = height * 100  # convert to cm
    if 30 <= weight <= 180 and 130 <= height <= 210:  # validation.
        BSA = ((height * weight) / 3600) * 0.5  # BSA formula.
    if 1.2 <= BSA <= 2.5:
        return BSA
    return 1.81  # return 1.81 means the BSA is unvalid so we return mean avg.

def get_RR_value(row):
    value = 0
    #for i in range(len(repo_file['h-num_demo'])):
    value = repo_file['value'][row]
    #break
    return value

def put_RR():
    value = 0
    for i in range(len(repo_file['h-num_demo'])):
        if vital_file['h-num_demo'][i] == repo_file['h-num_demo'][i]:
            value = repo_file['value'][i]
            worksheet.write(i + 1, 6, value)  # RR

def calculateCVRI(MAP, HR, RR, BSA):
    result_CVRI = 18 * MAP / (HR * RR * BSA)
    return result_CVRI


excel_file = 'vital demo.xlsx'
vital_file = pd.read_excel(excel_file)
excel_file = 'weight-height demo.xlsx'
WeightHeight_file = pd.read_excel(excel_file)
repo_file = pd.read_excel('resp demo.xlsx')  # reading RR file.

drug_num = []
all_the_first_time = []
all_the_last_time = []

row_num_first = []
row_num_last = []

first_id = vital_file['h-num_demo'][0]

first_drug = vital_file['para_code'][0]

row_num_first.append(2)
all_the_first_time.append(vital_file['time'][0])
drug_num.append(first_drug)

for i in range(len(vital_file['h-num_demo'])):
    if first_id == vital_file['h-num_demo'][i]:
        if first_drug != vital_file['para_code'][i]:
            first_drug = vital_file['para_code'][i]
            drug_num.append(first_drug)

            all_the_last_time.append(vital_file['time'][i - 1])
            all_the_first_time.append(vital_file['time'][i])

            row_num_first.append(i + 2)
            row_num_last.append(i + 1)

    else:
        row_num_last.append(i + 1)
        all_the_last_time.append(vital_file['time'][i - 1])

        total_time = []
        total_time_in_hours = []
        for j in range(len(all_the_first_time)):
            total_time.append(all_the_last_time[j] - all_the_first_time[j])

        for j in range(len(total_time)):
            total_time_in_hours.append(total_time[j].total_seconds() // 3600)

        # min_time = max(all_the_last_time)
        index_of_drug_we_need = []
        max_time = 0
        for d in drugs:
            indices = [i for i, x in enumerate(drug_num) if x == d]
            index_of_drug_we_need.append(indices)
            for index in indices:
                max_time = max(max_time, total_time_in_hours[index])
                # min_time = min(min_time, all_the_first_time[index])

        put_id(max_time, vital_file['h-num_demo'][i - 1])
        put_value(index_of_drug_we_need, row_num_first, row_num_last)

        row_excel += int(max_time)

        all_the_first_time.clear()
        all_the_last_time.clear()
        drug_num.clear()
        row_num_first.clear()
        row_num_last.clear()
        row_num_first.append(i + 2)
        first_id = vital_file['h-num_demo'][i]
        first_drug = vital_file['para_code'][i + 2]
        drug_num.append(first_drug)
        all_the_first_time.append(vital_file['time'][i])

#put_RR()
workbook.close()
'''
Using ffill method to fill the missing cell's with 
the prev value.
If the missing cell is the first one, we wont fill it.
'''


#def fill_fake_RR():
 #   for i in range(len(read_file['h-num_demo'])):
  #      repobook.write(i + 1,3,random.randint(4, 40))

#fill_fake_RR()

#repo.close()
read_file = pd.read_excel('new.xlsx')

read_file['MAP'] = read_file['MAP'].fillna(method='ffill')  # fill MAP with last value.
read_file['HR'] = read_file['HR'].fillna(method='ffill')  # fill HR with last value.
read_file['EVENT'] = read_file['EVENT'].fillna(method='ffill') #fill event.

for i in range(len(read_file['h-num_demo'])):
    read_file['RR'][i] = repo_file['value'][i]
    read_file['CVRI'][i] = calculateCVRI(read_file['MAP'][i], read_file['HR'][i], read_file['RR'][i], read_file['BSA'][i])

#read_file = read_file.iloc[:, 0:]

read_file.to_excel('new.xlsx')
'''
read_file = pd.read_excel('new.xlsx')

for i in range(len(read_file['h-num_demo'])):
    read_file['CVRI'][i] = calculateCVRI(read_file['MAP'][i],read_file['HR'][i],read_file['RR'][i],read_file['BSA'][i])

read_file.to_excel('new.xlsx')
'''