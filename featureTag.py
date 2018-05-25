import csv
import sys
import re

reload(sys)  
sys.setdefaultencoding('utf8')

def getTag():
    data = []
    sentence = 0
    with open('Tweet_token_tags - Sheet1.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        tmp = []
        for rows in reader:
            if(rows[0]):
                string = "sent: "+str(sentence)
                tmp.append(string)
                tmp.append(rows[0])
                if(rows[1]):
                    tmp.append(rows[1])
                else:
                    tmp.append("Other")
            else:
                tmp.append('')
                sentence += 1
            data.append(tmp)
            tmp = []
        

    csv_columns=["Sent", "Word", "Tag"]
    # writer = csv.DictWriter(ofile, csv_columns)
    myFile = open('taggedData.csv', 'w')  
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(csv_columns)
        writer.writerows(data)

    myFile.close()