# import necessary packages for mnist dataset training and testing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class KiNet_mlp:

       ## filters: a tuple of progressively larger filters so that our network learn more discriminate features
       ## regress: A boolean indicating whether or not a fully-connected linear activation will be added to the CNN for regression purpose
       
    def create_cnn(width, height, depth, filters = (2,4,8), regress = False):    
        
        # initialize the input shape and channel dimension assuming Tensorflow/ channels-last ordering (=> depth in the end)
        inputShape = (height, width, depth)
        chanDim = -1    

        # define the model input
        inputs = Input(shape = inputShape)    

        ## loop over the filters to create CONV => RELU => BN => POOL layers. Each iteration of the loop appends these layers

        for (i,f) in enumerate(filters): # f is the number of filters in each iteration

            # for the 1st CONV layer we set the input as x
            if i == 0:
                x = inputs

            # for rest iterations we progressively change x through the layers: CONV => RELU => BN => POOL
            x = Conv2D(f , (2,2), padding = "same")(x) # f is the number of filters in that particular layer: as example 16 filters/ kernels each with 3x3 convolutions to be done on input image x 
            x = Activation("relu")(x)
            x = BatchNormalization(axis = chanDim)(x)
            x = MaxPooling2D(pool_size = (2,2))(x)


        # Flatten the volume and then apply FC => RELU => BN => Dropout
        # x = Flatten() (x)
        x = GlobalAveragePooling2D() (x)  # GlobalAveragePooling2D(), compared to Flatten(), gave better accuracy values, and significantly reduced over-fitting and the no. of parameters.
        x = Dense (10000) (x)
        
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.4)(x)  # Dropout helps in reducing validation loss

        # Apply another FC layer to match the number of nodes coming out from the other model which is MLP 
        x = Dense(500)(x)
        x = Activation("relu")(x)

        # check to see if the regression node should be added
        if regress:
            x = Dense(1, activation ="linear")(x)

        # construct the CNN +> Finally here the model is constructed from inputs and all the layers of CNN that we have assembled together
        model = Model(inputs, x)    

        # return the CNN model
        model.summary()
        return model    


    def create_mlp(dim, regress = False):

	# define our MLP Network: architecture dims Here 1000 - 500

        model = Sequential()
        model.add(Dense(1000, input_dim = dim, activation = "relu"))
        
        # Make sure the below dimension within Dense is the same as the output dimension of create_cnn function
        model.add(Dense(500, activation = "relu"))

    	# check to see if the regression node is to be added
        if regress: # if we are performing regression, we add a Dense layer containing a single neuron with a linear activation function
            model.add(Dense(1, activation = "linear"))
        
        # return the mlp model
        model.summary()
        return model	   

