import os
dataDirList =os.listdir("//wsl.localhost/Ubuntu/root/fishyfinder/modded_fish")
print (dataDirList)

baseDir="//wsl.localhost/Ubuntu/root/fishyfinder"


trainData=os.path.join(baseDir,'train')
os.mkdir(trainData)

valData=os.path.join(baseDir,'validation')
os.mkdir(valData)

trainBSBData=os.path.join(trainData,'BSB')
os.mkdir(trainBSBData)

trainBSPData=os.path.join(trainData,'BSP')
os.mkdir(trainBSPData)

trainBRData=os.path.join(trainData,'BR')
os.mkdir(trainBRData)

trainCData=os.path.join(trainData,'C')
os.mkdir(trainCData)

trainGData=os.path.join(trainData,'G')
os.mkdir(trainGData)

trainJSData=os.path.join(trainData,'JS')
os.mkdir(trainJSData)

trainLSData=os.path.join(trainData,'LS')
os.mkdir(trainLSData)

trainLFData=os.path.join(trainData,'LF')
os.mkdir(trainLFData)

trainMData=os.path.join(trainData,'M')
os.mkdir(trainMData)

trainSNGFData=os.path.join(trainData,'SNGF')
os.mkdir(trainSNGFData)

trainSFCData=os.path.join(trainData,'SFC')
os.mkdir(trainSFCData)

trainSRData=os.path.join(trainData,'SR')
os.mkdir(trainSRData)

trainYFCData=os.path.join(trainData,'YFC')
os.mkdir(trainYFCData)

#VALIDATION
valBSBData=os.path.join(valData,'BSB')
os.mkdir(valBSBData)

valBSPData=os.path.join(valData,'BSP')
os.mkdir(valBSPData)

valBRData=os.path.join(valData,'BR')
os.mkdir(valBRData)

valCData=os.path.join(valData,'C')
os.mkdir(valCData)

valGData=os.path.join(valData,'G')
os.mkdir(valGData)

valJSData=os.path.join(valData,'JS')
os.mkdir(valJSData)

valLSData=os.path.join(valData,'LS')
os.mkdir(valLSData)

valLFData=os.path.join(valData,'LF')
os.mkdir(valLFData)

valMData=os.path.join(valData,'M')
os.mkdir(valMData)

valSNGFData=os.path.join(valData,'SNGF')
os.mkdir(valSNGFData)

valSFCData=os.path.join(valData,'SFC')
os.mkdir(valSFCData)

valSRData=os.path.join(valData,'SR')
os.mkdir(valSRData)

valYFCData=os.path.join(valData,'YFC')
os.mkdir(valYFCData)
