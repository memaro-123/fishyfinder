import os
import random
import shutil

splitSize=0.8

dataDirList =os.listdir("//wsl.localhost/Ubuntu/root/fishyfinder/modded_fish")
print (dataDirList)
def split_data(SOURCE,TRAINING,VALIDATION,SPLIT_SIZE):
    files=[]
    for filename in os.listdir(SOURCE):
        file = SOURCE +filename
        print(file)
        if os.path.getsize(file)>0:
            files.append(filename)
        else:
            print(filename + "-ignore")

    print(len(files))

    trainLength = int((len(files))*SPLIT_SIZE)
    validLength = (len(files) - trainLength)
    shuffedSet = random.sample(files, len(files))

    trainSet = shuffedSet[0:trainLength]
    valSet = shuffedSet[trainLength:]

    for filename in trainSet:
        thisfile=SOURCE+filename
        destination = TRAINING+filename
        shutil.copyfile(thisfile, destination)

    for filename in valSet:
        thisfile=SOURCE+filename
        destination = VALIDATION+filename
        shutil.copyfile(thisfile, destination)
        

split_data("","","","","","")