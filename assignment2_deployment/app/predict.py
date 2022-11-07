import numpy as np
import pandas as pd
import joblib

####### 
## Get the model trained in the notebook 
# `../nbs/1.0-asl-train_model.ipynb`
#######

model = joblib.load('models/simple_rf_model.joblib')


def preprocess(data):
    """
    Returns the features entered by the user in the web form. 

    To simplify, we set a bunch of default values. 
            For bools and ints, use the most frequent value
            For floats, use the median value

    Note that this represent some major assumptions that you'd 
    not want to make in real life. If you want to use default 
    values for some features then you'll have to think more 
    carefully about what they should be. 

    F.ex. if the user doesn't provide a value for BMI, 
    then one could use a value that makes more sense than 
    below. For example, the mean for the given gender would 
    at least be a bit more correct. 
    
    Having _dynamic defaults_ is important. And of course, if 
    relevant, getting some of the features without asking the user. 
    E.g. if the user is logged in and you can pull information 
    form a user profile. Or if you can compute or obtain the information 
    thorugh other means (e.g. user location if shared etc).
    """


    feature_values = {
        'MSSubClass': 56.897,
        'MSZoning': 'RL',
        'LotFrontage': 70.050,
        'LotArea': 10516.828,
        'Street': 'Pave',
        'Alley': 'NaN',
        'LotShape': 'Reg',
        'LandContour': 'Lvl',
        'Utilities': 'AllPub',
        'LotConfig': 'Inside',
        'LandSlope': 'LandSlope',
        'Neighborhood': 'NAmes',
        'Condition1': 'Norm',
        'Condition2': 'Norm',
        'BldgType': '1Fam',
        'HouseStyle': '1Story',
        'OverallQual': 6,
        'OverallCond':  5,
        'YearBuilt': 1971,
        'YearRemodAdd': 1985,
        'RoofStyle': 'Gable',
        'RoofMatl' : 'CompShg',
        'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd',
        'MasVnrType':'None',
        'ExterQual':'TA',
        'ExterCond': 'TA',
        'Foundation': 'PConc',
        'BsmtQual': 'TA',
        'BsmtCond': 'TA',
        'MasVnrArea': 103.685,
        'BsmtFinSF1':  443.640,
        'BsmtFinSF2': 46.549,
        'BsmtUnfSF ': 567.240,
        'TotalBsmtSF': 1057.429,
        '1stFlrSF': 1162.627,
        '2ndFlrSF': 346.992,
        'LowQualFinSF': 5.845, 
        'GrLivArea': 1515.464,
        'secondarydiagnosisnonicd9': 1,
        'BsmtFullBath': 0.425,
        'BsmtHalfBath': 0.058,
        'FullBath': 1.565,
        'HalfBath': 0.383,
        'BedroomAbvGr': 2.866,
        'KitchenAbvGr': 1.047,
        'TotRmsAbvGrd': 6.518,
        'Fireplaces': 1,
        'GarageYrBlt': 1979,
        'GarageCars': 2,
        'GarageArea': 472.980,
        'WoodDeckSF':  94.245,
        'OpenPorchSF': 46.660,
        'EnclosedPorch': 21.954,
        '3SsnPorch': 3.410,
        'ScreenPorch':  15.061,
        'PoolArea': 2.759,
        'MiscVal ': 43.489,
        'MoSold': 6,
        'YrSold': 2008,
    }


    # Parse the form inputs and return the defaults updated with values entered.

    for key in [k for k in data.keys() if k in feature_values.keys()]:
        feature_values[key] = data[key]

    return feature_values



####### 
## Now we can predict with the trained model:
#######


def predict(data):
    """
    If debug, print various useful info to the terminal.
    """
 
    # Store the data in an array in the correct order:

    column_order = ['rcount', 'gender', 'facid', 'eid', 'dialysisrenalendstage', 'asthma',
       'irondef', 'pneum', 'substancedependence', 'psychologicaldisordermajor',
       'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
       'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
       'creatinine', 'bmi', 'pulse', 'respiration',
       'secondarydiagnosisnonicd9']

    data = np.array([data[feature] for feature in column_order], dtype=object)


    # NB: In this case we didn't do any preprocessing of the data before 
    # training our random forest model (see the notebool `nbs/1.0-asl-train_model.ipynb`). 
    # If you plan to feed the training data through a preprocessing pipeline in your 
    # own work, make sure you do the same to the data entered by the user before 
    # predicting with the trained model. This can be achieved by saving an entire 
    # sckikit-learn pipeline, for example using joblib as in the notebook.
    
    pred = model.predict(data.reshape(1,-1))

    uncertainty = model.predict_proba(data.reshape(1,-1))

    return pred, uncertainty


def postprocess(prediction):
    """
    Apply postprocessing to the prediction. E.g. validate the output value, add
    additional information etc. 
    """

    pred, uncertainty = prediction

    # Validate. As an example, if the output is an int, check that it is positive.
    try: 
        int(pred[0]) > 0
    except:
        pass

    # Make strings
    pred = str(pred[0])
    uncertainty = str(uncertainty[0])


    # Return
    return_dict = {'pred': pred, 'uncertainty': uncertainty}

    return return_dict