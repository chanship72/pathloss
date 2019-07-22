import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from equationmodel_ann import ann_learning

def ADD_data_loader(fileNameDic):
    columnLabel = ["type", "distance", "pathloss", "heightTM"]
    combinedDataList = []
    for fName, constTuple in fileNameDic.items():
        df = dataLoaderCSV(fName, columnLabel)
        # add frequency feature
        # log frequency = log10(freq)
        print("distance filtering(before):", df.count())
        if "iksan" in fName:
            df = filteringDF(df, 'distance', [0, 3])
        if "nonsan" in fName:
            df = filteringDF(df, 'distance', [0, 2.3])
        if "paju" in fName:
            df = filteringDF(df, 'distance', [0, 4])
        print("distance filtering(after):", df.count())
        for (label, value) in constTuple:
            df = addFeatureWithConst(df, value, label)
        combinedDataList.append(df)
    combinedDataFrame = pd.concat(combinedDataList)

    # filtering moving data
    combinedDataFrame = combinedDataFrame[combinedDataFrame.type == 'm']

    combinedDataFrame['heightTratio'] = combinedDataFrame['heightTM'] / combinedDataFrame['heightTB']
    combinedDataFrame['freePathloss'] = getFreeSpacePathLoss(combinedDataFrame['distance'], combinedDataFrame['frequency'])

    # distance conversion: distance(KM)-> log10(distance)
    combinedDataFrame['logDistance'] = convertlogarithm(combinedDataFrame, ['distance'])
    # antenna height conversion : KM -> M
    combinedDataFrame['heightTM'] = convertKM(combinedDataFrame, ['heightTM'])
    # log frequency: frequency(Mhz) -> log10(frequency)
    combinedDataFrame['logFrequency'] = convertlogarithm(combinedDataFrame, ['frequency'])
    # log antenna height B(transmitter): heightB(meter) -> log10(heightB)
    combinedDataFrame['logHeightB'] = convertlogarithm(combinedDataFrame, ['heightB'])
    # log antenna height M(receiver): heightM(meter) -> log10(heightM)
    combinedDataFrame['logHeightM'] = convertlogarithm(combinedDataFrame, ['heightM'])
    # log antenna ration(heightTM/heightTB): heightTM/heightTB(meter) -> log10(heightTM/heightTB)
    combinedDataFrame['logHeightTratio'] = convertlogarithm(combinedDataFrame, ['heightTratio'])
    # logh_b * logd term
    combinedDataFrame['logAntennaMulLogDistance'] = combinedDataFrame['logHeightB'] * combinedDataFrame['logDistance']

    return combinedDataFrame

def filteringDF(df, col, valueRange):
    return df[(df[col] >= valueRange[0]) & (df[col] <= valueRange[1])]

def combineDF(dfList):
    return pd.concat(dfList)

def dataLoaderCSV(fileName, colLabel):
    return pd.read_csv(fileName, delimiter=',', names=colLabel)

def convertKM(dataFrame, targetColName = 'dist', direction=True):
    # convert distance unit (KM <-> M)
    # direction(Boolean):
    # - True: KM -> M
    # - False: M -> KM
    if direction:
        return dataFrame[targetColName] * 1000
    else:
        return dataFrame[targetColName] / 1000.0

def addFeatureWithConst(dataframe, val, label):
    dataframe[label] = val
    return dataframe

def convertlogarithm(dataFrame, targetColumn):
    # convert data into logarithmic term
    # targetColIdx: taget column/feature index
    return dataFrame[targetColumn].apply(np.log10, axis = 1)

def getFreeSpacePathLoss(distance, frequency):
    return 20*np.log10(distance)+20*np.log10(frequency) + 32.45

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def fillColwithConstant(df, colName, val):
    df[colName] = val
    return df

def splitDFwithCol(df, colName, valList):
    res = []
    for val in valList:
        res.append(df[df[colName] == val])
    return res

def makeXforGraph(X, Y, category):
    # addData[['logDistance', 'logFrequency', 'logHeightB', 'logHeightM', 'logHeightTratio', 'logAntennaMulLogDistance']]
    tmpCombined = pd.concat([X,Y.reindex(X.index)], axis=1)

    convertedX, convertedY = [], []
    groupDF = [x for _, x in tmpCombined.groupby(['logFrequency'])]

    for df in groupDF:
        meanDF = df.mean(axis=0)

        tmpDF = fillColwithConstant(df, 'logHeightB', meanDF['logHeightB'])
        tmpDF = fillColwithConstant(df, 'logHeightM', meanDF['logHeightM'])
        tmpDF = fillColwithConstant(df, 'logHeightTratio', meanDF['logHeightTratio'])
        tmpDF = fillColwithConstant(df, 'logAntennaMulLogDistance', meanDF['logAntennaMulLogDistance'])
        
        # print("tmpDF:",tmpDF)

        convertedX.append(tmpDF[['logDistance', 'logFrequency', 'logHeightB', 'logHeightM', 'logHeightTratio', 'logAntennaMulLogDistance']])
        convertedY.append(tmpDF['pathloss'])

    return [convertedX, convertedY]




