import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
            df = filteringDF(df, 'distance', [1.0, 3.0])
        if "nonsan" in fName:
            df = filteringDF(df, 'distance', [1.0, 2.3])
        if "paju" in fName:
            df = filteringDF(df, 'distance', [1.258925, 4.0])
        distanceFilteredTotalCount += df.size
        print("{}: distance filtering(after):{}".format(fName,df.shape))
        # adding constant features
        for (label, value) in constTuple:
            df = addFeatureWithConst(df, value, label)
        combinedDataList.append(df)
    combinedDataFrame = pd.concat(combinedDataList, axis=0, keys=columnLabel, sort=False)
    print(combinedDataFrame.head())
    print("Combined data set:", combinedDataFrame.shape)
    
    # filtering moving data
    print("type filtering(before):{}".format(combinedDataFrame.shape))
    combinedDataFrame = combinedDataFrame[combinedDataFrame.type == 'm']
    print("type filtering(after):{}".format(combinedDataFrame.shape))
    
    # term creation
    print("Dataframe before add new terms:",combinedDataFrame.shape)
#     # distance conversion: distance(KM)-> distance(M)
#     combinedDataFrame['distance'] = convertKM(combinedDataFrame, ['distance'])
    # free pathloss: distance(m), frequency(mhz)
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
    combinedDataFrame['logHeightTM'] = convertlogarithm(combinedDataFrame, ['heightTM'])
    # log antenna ration(heightTM/heightTB): heightTM/heightTB(meter) -> log10(heightTM/heightTB)
    combinedDataFrame['logHeightTratio'] = convertlogarithm(combinedDataFrame, ['heightTratio'])
    # log extended antenna ration(heightTM/heightTB): (height_TM + height_M) / (height_TB + height_B)(meter) -> log10((height_TM + height_M) / (height_TB + height_B))
    combinedDataFrame['logExtendedHeightTratio'] = abs(convertlogarithm(combinedDataFrame, ['extendedHeightTratio']))
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
    return 20*np.log10(distance)+20*np.log10(frequency) + 32.45 #- 27.55(M)

def mlp_prediction_error(model, X, Y):
    X_predictions = model.predict(X)
    rmse = np.sqrt(np.mean(np.power(Y-X_predictions,2)))
    
    return rmse
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

def samplingData(df, percentage, weight, prFlag = True, randomState = 1):
    if prFlag:
        print("data distribution(before)")
        print(df.describe())

        print("sampling shape(before):{}".format(df.shape))
    dfSample = df.sample(frac=percentage, replace=True, random_state=randomState, weights=weight)

    if prFlag:
        print("sampling shape(after):{}".format(dfSample.shape))

        print("data distribution(after)")
        print(dfSample.describe())    

    return dfSample

def normalizeData(X, Y, scaler = 'standard', prFlag=True):
    # name: Normalize Data
    # author: cspark 
    # @param df: dataframe
    # @param scaler = standard, minmax, manual
    # @return scaled dataframe
    # @return scaler class (for inverseTransformation)
    
#            logDistance  logFrequency  logHeightB  logHeightM  logExtendedHeightTratio  logHeightTratio  logAntennaMulLogDistance  pathloss
# count     79125.00      79125.00    79125.00    79125.00                 79125.00         79125.00                  79125.00  79125.00
# mean          0.31          3.03        1.05        0.30                    -0.40            -0.32                      0.32    127.47
    
    if prFlag:
        print("normalization distribution(before):\n{}".format(X.describe()))
    scaledData = None
    dataArray = np.array(X)
    y = np.array(Y)
    if scaler == 'standard':
        scaler = StandardScaler()
        scaledData = scaler.fit_transform(dataArray)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        scaledData = scaler.fit_transform(dataArray)
    elif scaler == 'manual':
        # manually
        if 'logFrequency' in X.columns:
            X.loc[:,'logFrequency'] *= 0.1
        if 'logHeightB' in X.columns:
            X.loc[:,'logHeightB'] *= 0.1
        scaledData = np.array(X)

    dfNormalized = pd.DataFrame(scaledData, columns=X.columns)
    if prFlag:    
        print("normalization distribution(after):\n{}".format(dfNormalized.describe()))
    
    return dfNormalized, scaler

def makeXforGraphWithGroupingSeason(X, Y):
    tmpCombined = pd.concat([X,Y.reindex(X.index)], axis=1)

    convertedX, convertedY = [], []
    groupDF = {k: v for k, v in tmpCombined.groupby('season')}
    for season, df in groupDF.items():
        convertedX.append(df[X.columns])
        convertedY.append(df[Y.columns])
        
    return [convertedX, convertedY]

def makeXforGraphWithGroupingFrequency(X, Y, excludedCols):
    tmpCombined = pd.concat([X,Y.reindex(X.index)], axis=1)

    convertedX, convertedY = [], []
    groupDF = {k: v for k, v in tmpCombined.groupby('logFrequency')}
    for freq, df in groupDF.items():
        meanDF = df.mean(axis=0)
        print(meanDF)
        for col in excludedCols:
            df = fillColwithConstant(df, col, meanDF[col])
        
#         print("Group-{:6.2f}Ghz Data shape:{}".format(df['logFrequency'].iloc[0],df.shape))

        convertedX.append(df[X.columns])
        convertedY.append(df[Y.columns])
        
    return [convertedX, convertedY]

def makeXforGraph(X, Y, removeCols = ['logDistance', 'logFrequency']):
    tmpCombined = pd.concat([X,Y.reindex(X.index)], axis=1)
    excludedCols = list(tmpCombined.columns.values)
    for col in removeCols:
        if col in excludedCols:
            excludedCols.remove(col)
    
    meanDF = tmpCombined.mean(axis=0)
    print(meanDF)
    tmpDF = tmpCombined
    for col in excludedCols:
        tmpDF = fillColwithConstant(tmpDF, col, meanDF[col])
    
    return [tmpDF[X.columns], tmpDF[Y.columns]]

def groupWithFrequency(X, Y, orderCol):
    tmpCombined = pd.concat([X,Y.reindex(X.index)], axis=1)

    convertedX, convertedY = [], []
    groupDF = {k: v for k, v in tmpCombined.groupby('logFrequency')}
    for freq, df in groupDF.items():
        df = df.sort_values(by=[orderCol])
        convertedX.append(df[X.columns])
        convertedY.append(df[Y.columns])
        
    return [convertedX, convertedY]

def inverseScale(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaledData = scaler.inverse_transform(data)
    print(scaledData)
    
    return scaledData

def train_2d_graph(model, linearModel, originLinearModel, X, Y, targetCol, targetColLabel, xCategory = ('0.4Ghz', '1.399Ghz', '2.249Ghz'), convertFlag = True):
    #   @param X: list of dataframe [df1, df2, ...] Grouped by category
    #   @param Y: list of dataframe [df1, df2, ...]
    #   @param targetColX: list of target column of dataframe ['logDistance', 'logAntennaMulLogDistance']
    #   @param xCategory: xlabel

    fig,ax = plt.subplots(3, 1)
    fig.set_figwidth(8)
    fig.set_figheight(16)
    cmap = plt.cm.coolwarm
    cmap_i = 0.0

    for idx in range(len(X)):
        idxCol = X[idx].columns.get_loc(targetCol)
        Xscatter = np.array(X[idx])[:,idxCol]
        Yscatter = np.array(Y[idx])
        
        minVal = X[idx][targetCol].min()
        maxVal = X[idx][targetCol].max()
        linX = np.linspace(minVal, maxVal, num=len(np.array(X[idx])))
        
        arr = np.array(X[idx])
        if convertFlag:
            arr[:,idxCol] = linX
        
        pred = model.predict(arr)
        ax[idx].set_title(xCategory[idx])
        ax[idx].scatter(Xscatter, Yscatter, s=1, label='data') 
        ax[idx].plot(linX, pred, color=cmap(cmap_i), label='ANN training')
        if linearModel:
            pred_linear = linearModel.predict(arr)
            ax[idx].plot(linX, pred_linear, dashes=[6, 2], color=cmap(cmap_i), label='Multivariate Linear Model')
        if originLinearModel:
            originLinearModel = Ridge(alpha=0.0001)
            originLinearModel.fit(Xscatter.reshape(-1,1), Yscatter)
            print("original_L_pathloss = {:6.2f}log_d + {:6.2f}".format(originLinearModel.coef_[0],originLinearModel.intercept_))

            pred_origin = originLinearModel.predict(arr[:,0].reshape(-1,1))
            ax[idx].plot(linX, pred_origin, dashes=[2, 4], color=cmap(cmap_i), label='Original Linear Model')
            
        cmap_i += 0.8

        ax[idx].set_xlabel(targetColLabel)
        ax[idx].set_ylabel("Path Loss(dB)")
        ax[idx].legend()
    plt.subplots_adjust(hspace=0.4)
#     plt.legend(xCategory)
    plt.show()

def train_2d_sigma_graph(model, X, Y, targetCol = 'logDistance', xCategory = ('Winter', 'Spring', 'Summer'), sigmaFlag = True):
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
        minXlogD = X[idx][targetCol].min()
        maxXlogD = X[idx][targetCol].max()

        linXlogD = np.linspace(minXlogD, maxXlogD, num=len(np.array(X[idx])))
        X[idx][targetCol] = linXlogD
        elementX = np.array(X[idx])
        elementY = np.array(Y[idx])
        
        if sigmaFlag:
            pred, sigma = model.predict(elementX, return_std=True)       
            plt.plot(elementX[:,0], sigma, color=cmap(cmap_i))
        else:
            pred = model.predict(elementX)       
            plt.plot(elementX[:,0], pred, color=cmap(cmap_i))            
        cmap_i += 0.8

    plt.xlabel("log distance(KM)")
    plt.ylabel("Standard Deviation(dB)")
    plt.legend(xCategory)
    plt.show()    

def train_2d_sigma_graph_s(model, X, Y, targetCol = 'logDistance', xLabel= "log distance(KM)", sigmaFlag = True):
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

    minXlogD = min(X)+0.001
    maxXlogD = max(X)-0.001
    linXlogD = np.linspace(minXlogD, maxXlogD, num=len(X)).reshape(-1,1)
        
    if sigmaFlag:
        pred, sigma = model.predict(linXlogD, return_std=True)       
        plt.plot(linXlogD, sigma, color=cmap(cmap_i))
        plt.ylabel("Standard Deviation(dB)")
    else:
        pred = model.predict(linXlogD)       
        plt.plot(linXlogD, pred, color=cmap(cmap_i))            
        plt.ylabel("Path Loss(dB)")
    cmap_i += 0.8

    plt.xlabel(xLabel)

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

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma', edgecolor='none')
    # ax.plot_trisurf(x, y, z ,cmap='binary', alpha=0.5)
    # ax.contour3D(x, y, z, 50, cmap='binary')


    ax.set_xlabel(xlabel,labelpad=18,fontsize=18)
    ax.set_ylabel(ylabel,labelpad=18,fontsize=18)
    ax.set_zlabel(zlabel,labelpad=10,fontsize=18)
    ax.view_init(elev=30, azim=220)

    plt.minorticks_on()
    plt.rcParams['xtick.labelsize']=15
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


    plt.show()
    