import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# from equationmodel_ann import ann_learning

def ADD_data_loader(fileNameDic):
    print("ADD data preprocessing")
    columnLabel = ["type", "distance", "pathloss", "heightTM"]
    combinedDataList = []
    totalCount, distanceFilteredTotalCount = 0, 0
    for fName, constTuple in fileNameDic.items():
        df = dataLoaderCSV(fName, columnLabel)
        # distance filtering
        totalCount+=df.size
        print("{}: distance filtering(before):{}".format(fName,df.shape))
        if "iksan" in fName:
            df = filteringDF(df, 'distance', [1, 3])
        if "nonsan" in fName:
            df = filteringDF(df, 'distance', [1, 3])
        if "paju" in fName:
            df = filteringDF(df, 'distance', [1, 3])
        distanceFilteredTotalCount += df.size
        print("{}: distance filtering(after):{}".format(fName,df.shape))
        # adding constant features
        for (label, value) in constTuple:
            df = addFeatureWithConst(df, value, label)
        combinedDataList.append(df)
    combinedDataFrame = pd.concat(combinedDataList, keys=columnLabel, sort=True)
    print("Combined data set:", combinedDataFrame.shape)
    
    # filtering moving data
    print("type filtering(before):{}".format(combinedDataFrame.shape))
    combinedDataFrame = combinedDataFrame[combinedDataFrame.type == 'm']
    print("type filtering(after):{}".format(combinedDataFrame.shape))
    
    # term creation
    print("Dataframe before add new terms:",combinedDataFrame.shape)
    combinedDataFrame['freePathloss'] = getFreeSpacePathLoss(combinedDataFrame['distance'], combinedDataFrame['frequency'])
    # distance conversion: distance(KM)-> log10(distance)
    combinedDataFrame['logDistance'] = convertlogarithm(combinedDataFrame, ['distance'])
    # antenna height conversion : KM -> M
    combinedDataFrame['heightTM'] = convertKM(combinedDataFrame, ['heightTM'])
    # height T ratio = height_TM / height_TB
    combinedDataFrame['heightTratio'] = combinedDataFrame['heightTM'] / combinedDataFrame['heightTB']
    # extended height T ratio = (height_TM + height_M) / (height_TB + height_B)
    combinedDataFrame['extendedHeightTratio'] = (combinedDataFrame['heightTM'] + combinedDataFrame['heightM']) / (combinedDataFrame['heightTB'] + combinedDataFrame['heightB'])    
    # log frequency: frequency(Mhz) -> log10(frequency)
    combinedDataFrame['logFrequency'] = convertlogarithm(combinedDataFrame, ['frequency'])
    # log antenna height B(transmitter): heightB(meter) -> log10(heightB)
    combinedDataFrame['logHeightB'] = convertlogarithm(combinedDataFrame, ['heightB'])
    # log antenna height M(receiver): heightM(meter) -> log10(heightM)
    combinedDataFrame['logHeightM'] = convertlogarithm(combinedDataFrame, ['heightM'])
    # log antenna ration(heightTM/heightTB): heightTM/heightTB(meter) -> log10(heightTM/heightTB)
    combinedDataFrame['logHeightTratio'] = convertlogarithm(combinedDataFrame, ['heightTratio'])
    # log extended antenna ration(heightTM/heightTB): (height_TM + height_M) / (height_TB + height_B)(meter) -> log10((height_TM + height_M) / (height_TB + height_B))
    combinedDataFrame['logExtendedHeightTratio'] = convertlogarithm(combinedDataFrame, ['extendedHeightTratio'])
    # logh_b * logd term
    combinedDataFrame['logAntennaMulLogDistance'] = combinedDataFrame['logHeightB'] * combinedDataFrame['logDistance']
    # (log distance + log distance * log_b)
    combinedDataFrame['combinedDistance'] = (combinedDataFrame['logDistance'] + combinedDataFrame['logHeightB'] * combinedDataFrame['logDistance'])
    print("Dataframe after add constant feature:",combinedDataFrame.shape)
    
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

def samplingData(df, percentage):
    print("data distribution(before)")
    print(df.describe())

    print("sampling shape(before):{}".format(df.shape))
    dfSample = df.sample(frac=percentage, replace=False, random_state=1)
    print("sampling shape(after):{}".format(dfSample.shape))

    print("data distribution(after)")
    print(dfSample.describe())    

    return dfSample

def normalizeData(df, scaler = 'standard', auto = True):
    print("normalization distribution(before):\n{}".format(df.describe()))
    scaledData = None
    if auto:
        dataArray = np.array(df)

        if scaler == 'standard':
            scaler = StandardScaler()
        elif scaler == 'norm-l1':
            scaler = Normalizer(norm='l1')
        elif scaler == 'norm-l2':
            scaler = Normalizer(norm='l1')
        else:
            scaler = StandardScaler()
        scaledData = scaler.fit_transform(dataArray)

    else:
        # manually
        df.loc[:,'logFrequency'] *= 0.1
        df.loc[:,'logHeightB'] *= 0.1
        scaledData = np.array(df)
    
    dfNormalized = pd.DataFrame(scaledData, columns=df.columns)
    print("normalization distribution(after):\n{}".format(dfNormalized.describe()))
    
    return scaledData

def makeXforGraphWithGroupingFrequency(X, Y, excludedCols = ['logHeightB', 'logHeightM', 'logHeightTratio']):
    tmpCombined = pd.concat([X,Y.reindex(X.index)], axis=1)

    convertedX, convertedY = [], []
    groupDF = {k: v for k, v in tmpCombined.groupby('logFrequency')}
    for freq, df in groupDF.items():
        meanDF = df.mean(axis=0)
        
        for col in excludedCols:
            df = fillColwithConstant(df, col, meanDF[col])
        
        print("Group-{:6.2f}Ghz Data shape:{}".format(df['logFrequency'].iloc[0],df.shape))

        convertedX.append(df[X.columns])
        convertedY.append(df[Y.columns])
        
    return [convertedX, convertedY]

def makeXforGraph(X, Y, targetCols = ['logDistance', 'logFrequency']):
#     print("X shape:{}, Y shape:{}".format(X.shape, Y.shape))
    tmpCombined = pd.concat([X,Y.reindex(X.index)], axis=1)
#     print("tmpCombined(before):{}".format(tmpCombined.shape))
#     print("tmpCombined\n",tmpCombined.head())
    excludedCols = list(tmpCombined.columns.values)
#     print(excludedCols)
    for col in targetCols:
        excludedCols.remove(col)
#     print(excludedCols)
    
    meanDF = tmpCombined.mean(axis=0)
#     print("meanDF:\n",meanDF)
    tmpDF = tmpCombined
#     print("excludedCols:{}".format(excludedCols))
    for col in excludedCols:
        tmpDF = fillColwithConstant(tmpDF, col, meanDF[col])
        
#     print("tmpCombined(after):{}".format(tmpDF.shape))
#     print(tmpDF.head())
    
    return [tmpDF[X.columns], tmpDF[Y.columns]]

def inverseScale(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaledData = scaler.inverse_transform(data)
    print(scaledData)
    
    return scaledData

def train_2d_graph(model, X, Y, targetColX, xCategory = ('0.4Ghz', '1.399Ghz', '2.249Ghz')):
    #   @param X: list of dataframe [df1, df2, ...] Grouped by category
    #   @param Y: list of dataframe [df1, df2, ...]
    #   @param targetColX: list of target column of dataframe ['logDistance', 'logAntennaMulLogDistance']
    #   @param xCategory: xlabel

    fig,ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(6)
    cmap = plt.cm.coolwarm
    # print("X:",X)
    # print("Y:",Y)
    cmap_i = 0.0

    for idx in range(len(X)):        
        for col in targetColX:        
            minX = X[idx][col].min()
            maxX = X[idx][col].max()
            linX = np.linspace(minX, maxX, num=len(np.array(X[idx])))
            print("For '{}' column, min value:{:6.2f}, max value:{:6.2f}".format(col, minX, maxX))

            X[idx][col] = linX
        
        elementX = np.array(X[idx])
        elementY = np.array(Y[idx])

        pred = model.predict(elementX)        
        plt.plot(elementX[:,0], pred, color=cmap(cmap_i))
        cmap_i += 0.8

    plt.xlabel("log distance(KM)")
    plt.ylabel("Path Loss(dB)")
    plt.legend(xCategory)
    plt.show()
    
def train_3d_graph(model, X, Y, targetColX, xlabel = "Log distance(m)", ylabel = "Frequency(Ghz)", zlabel = "Path Loss(dB)"):
    #   @param X: list of dataframe [df1, df2, ...] Grouped by category
    #   @param Y: list of dataframe [df1, df2, ...]
    #   @param targetColX: list of target column of dataframe ['logDistance', 'logAntennaMulLogDistance']
    #   @param xCategory: xlabel

    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)
       
    totalPoint = 100
    linXList = []
    for targetCol in targetColX:        
        minX = X[targetCol].min()
        maxX = X[targetCol].max()
        linX = np.linspace(minX, maxX, num=totalPoint)
#         print("For '{}'' column, min value:{:6.2f}, max value:{:6.2f}".format(targetCol, minX, maxX))

        linXList.append(linX)

    x = np.array(linXList[0])
    y = np.array(linXList[1])

    x, y = np.meshgrid(x, y)
    xFlat, yFlat = x.flatten(), y.flatten()
    df = pd.DataFrame(xFlat, columns=['dummy'])
    print(X.columns)
    for c in X.columns:
        if c == targetColX[0]:
            df[c] = xFlat
        elif c == targetColX[1]:
            df[c] = yFlat
        else:
            val = X[c].iloc[0]
            df = addFeatureWithConst(df, val, c)
    df.drop('dummy', axis=1, inplace=True)
    print("input data/feature distribution:\n",df.describe())
    newX = np.array(df)

    z = model.predict(newX)
    z = z.reshape(x.shape)    
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='binary', edgecolor='none')
    # ax.plot_trisurf(x, y, z ,cmap='binary', alpha=0.5)
    # ax.contour3D(x, y, z, 50, cmap='binary')
    
    ax.set_xlabel(xlabel,labelpad=18,fontsize=18)
    ax.set_ylabel(ylabel,labelpad=18,fontsize=18)
    ax.set_zlabel(zlabel,labelpad=10,fontsize=18)
    ax.view_init(elev=20, azim=220)

    plt.minorticks_on()
    plt.rcParams['xtick.labelsize']=15
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    