from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator,HillClimbSearch,BayesianEstimator
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

##Bayesian NetWork Creation
def bay_net_structure(data,k):
    #data_2d=data.reshape(-1, 6)
    #data_2d=pd.DataFrame(data,columns=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z'])
    data=data[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z','label']]
    hc = HillClimbSearch(data[0:k])
    best_model = hc.estimate()

    print("Edges of the learned Bayesian network:")
    print(best_model.edges())


#Random Forest Algorithm
def random_forest(data,labels):
    x=data
    y=labels
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    rf = RandomForestClassifier(criterion='entropy',min_samples_leaf=50)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    for i in range(3):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(tree,
                                feature_names=X_train.columns,  
                                filled=True,  
                                max_depth=2, 
                                impurity=False, 
                                proportion=True)
        graph = graphviz.Source(dot_data)
        return graph


#2D Convolutional Neural Network
def nn(data,labels):
    x=data
    y=labels
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(X_train[0].shape)
    print(y_train.shape)
   
    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(16, activation='softmax'))
        

    
    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    history = model.fit(X_train, y_train,batch_size=32, epochs = 10, validation_data= (X_test, y_test), verbose=1)
    return model, history