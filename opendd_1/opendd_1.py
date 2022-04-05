"""
opendd_1 is basically opendd v1 which stands for opensource drug discovery v1

It's a simple software interface where data integration could be performed with the models along with required preprocessing, machine learning training, inference and visualization to
predict critical drug properties, identification and screening for candidate drug molecules. 

opendd_1 shows a simple case study pertaining to inhibition constant (Ki) predictions of candidate drug molecules to Human Coagulation Factor X (FX) protein as obtained from chembl database. 

"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import os
import numpy as np

from tensorflow.keras.models import load_model


from input_gui_train_diff_opendD_1 import CNN_MLP_train_test

from chembl_drug_prediction.chembl_descriptor_generation_Ki import chembl_descriptor_generation


import tkinter.messagebox
from tkinter import simpledialog

global scaling_factor_diff




##---------------------molpredv1.0 LOGO placement on root canvas--------------------------------------------------------------------------------------------------------------

# Initiate the browser which holds the image displays
root = tk.Tk()
root.title('molpredv1.0')

#Define the canvas window and tether it using grid for the image displays
canvas = tk.Canvas(root, width = 500, height = 500)
# root.geometry("800x600")
canvas.grid(columnspan = 6, rowspan = 6)


# put the logo
logo = Image.open('molpred_logo1.PNG')
logo = ImageTk.PhotoImage(logo)
# canvas.create_image(200,100,image=logo)

logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column = 3, row = 6, padx = 200, pady = 100)

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------.....



##---------------------user option to open an image file which is resized and displayed on root canvas--------------------------------------------------------------------------

# Put an instruction to select image to display
instructions = tk.Label(root, text = "View image", font = "Arial")
instructions.grid( column = 0, row = 0)

# define a file browsing function
def open_image_file():

    browse_text.set("loading...")

    filepath = filedialog.askopenfilename(parent = root, initialdir = os.getcwd(),title = "select a file and double click on it to import", filetypes = [("img files", "*.png")])


    if filepath: # i.e. if filepath == True:
        image = Image.open(filepath)
        image = image.copy()
        # optional
        # image = resize_image(image)

        image = ImageTk.PhotoImage(resize_image(image))
        image_label = tk.Label(image=image)
        # essential
        image_label.image = image
        image_label.grid(column = 0, row = 3)

        # add a text regarding image description
        text = tk.Label(root, height = 2, width = 50, text = "Resized 300 x 300 pixel image of the chosen molecule", font = "Arial")
        text.grid(column = 1, row = 3)

    return filepath        


# resize the displayed image
def resize_image(image):

    new_height = 300;
    new_width = 300;
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return image

# setup a browse button implementing open_image_file() which displays an image by browsing 
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable = browse_text, command = lambda: open_image_file(), font = "Arial", bg = "#20bebe", fg = "white", height = 2, width = 15)
browse_text.set("Browse image")
browse_btn.grid(column = 0, row =1)

##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------........................................................................



##--------------------------------------------------------------------Generating the required dataset-------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Put an instruction to select image to display
instructions = tk.Label(root, text = "Load .csv data", font = "Arial")
instructions.grid( column = 0, row = 2)

# define a file browsing function
def open_csv_file():

    browse_text.set("loading...")

    filepath = filedialog.askopenfilename(parent = root, initialdir = os.getcwd(),title = "select a file and double click on it to import", filetypes = [("csv files", "*.csv")])


    if filepath: # i.e. if filepath == True:
        # add a text regarding image description
        text = tk.Label(root, height = 2, width = 50, text = "The input data file containing SMILES is loaded", font = "Arial")
        text.grid(column = 0, row = 5)
        
    print ("The data filepath is: ", filepath)    

    return filepath


# Required dataset collecton and synthesis using the .csv file
def data_processing(open_csv_file):
    
    status = chembl_descriptor_generation(open_csv_file)    
    
    return status
    

# setup a browse button implementing open_csv_file() which selects the raw csv data file by browsing 
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable = browse_text, command = lambda: data_processing(open_csv_file()), font = "Arial", bg = "#10bebe", fg = "yellow", height = 2, width = 15)
browse_text.set("Browse csv")
browse_btn.grid(column = 0, row = 3)



##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      

##..................................................... Set of functions to collect hyperparameters from user input using buttons in canvas roo2 and then run the respective models to predict variables................................................................
##------------------------------------------------------ hyperparameters: lr_tuning, decay_tuning, bs_tuning, epochs_tuning, momemntum_tuning, dropout_val_tuning--------------------------------------------------------------------------------------------------------
##------------------------------------------------------ global dictionary: define a global dicionary contaninig all hyperparameters for communication across different roots...........................................................................................
## ----------------------------------------------------- make sure that all hyperparameters are global variables--------------------------------------------------------------------------------------------------------------------------------------------------------

# # # # # instantiate an empty parameter dictionary
# global hyperparameter_dict
# hyperparameter_dict = {}

def hyperparameter_lr(root2):
    lr_tuning =  float(lr.get())
    hp.append(lr_tuning)
    

    mytext=tk.Label(root2,text = "Input learning rate is:  " + str(lr_tuning), font = "Arial")
    mytext.grid(column = 0, row =1)

    print("Hyperparameter lr:", lr_tuning)
    return lr_tuning

def hyperparameter_decay(root2):
    decay_tuning =  float(decay.get())
    hp.append(decay_tuning) 
    

    mytext=tk.Label(root2,text = "Input decay rate is:  " + str(decay_tuning), font = "Arial")
    mytext.grid(column = 0, row =2)

    print("Hyperparameter decay:", decay_tuning )
    return decay_tuning  

def hyperparameter_bs(root2):
    bs_tuning =  float(bs.get())
    hp.append(bs_tuning)


    mytext=tk.Label(root2,text = "Input batch size is:  " + str(bs_tuning), font = "Arial")
    mytext.grid(column = 0, row =3)

    print("Hyperparameter bs:", bs_tuning )
    return bs_tuning  

def hyperparameter_epochs(root2):
    epochs_tuning =  float(epochs.get())
    hp.append(epochs_tuning)


    mytext=tk.Label(root2,text = "Input # of epochs is:  " + str(epochs_tuning), font = "Arial")
    mytext.grid(column = 0, row =4)

    print("Hyperparameter epochs:", epochs_tuning )
    return epochs_tuning   

def hyperparameter_momentum(root2):
    momentum_tuning =  float(momentum.get())
    hp.append(momentum_tuning)
    

    mytext=tk.Label(root2,text = "Input momentum is:  " + str(momentum_tuning), font = "Arial")
    mytext.grid(column = 0, row =5)

    print("Hyperparameter momentum:", momentum_tuning )
    return momentum_tuning   

def hyperparameter_dropout(root2):
    dropout_tuning =  float(dropout.get())
    hp.append(dropout_tuning)


    mytext=tk.Label(root2,text = "Input dropout is:  " + str(dropout_tuning), font = "Arial")
    mytext.grid(column = 0, row =6)

    print("Hyperparameter dropout:", dropout_tuning )
    return dropout_tuning    

# This function checks whether the hyperparameter dictionary is available to other functions (such as this function itself)
def check_dictionary( hyperparameter_dict ):
    print("The data dictionary available:", hyperparameter_dict )


def parameter_selection():
    global check
    # accessing the parameter string using click.get()
    if click.get() == "inhibition constant (Ki)" or " " :
        drop_label_return = click.get()
        print("Selected parameter:", drop_label_return)

    check = drop_label_return    

    return check

def model_selection():
    global check_model
    # accessing the parameter string using click2.get(). The parameters are listed in "xx" options array below
    if click2.get() == "CNN & MLP hybrid model" or " ":
        drop_label_return2 = click2.get()
        print("Selected model:", drop_label_return2)

    check_model = drop_label_return2    


    drop_label2 = Label(root, text = drop_label_return2 + ' model has been activated to use')

    # Basically showing the result of a click (on OptionMenu): Showing the selected model
    drop_label2.grid(column = 4, row =3)

    return check_model   


# Collecting all the user inputs and feeding them into training python file: from input_gui_train_diff import diff_train
def collect_parameters(root2, root):


    # This parameter dictionary (hyperparameter_dict) should be accessible under/from any function
    ## Store function as dictionary values accessible to other functions
    hyperparameter_dict = {"lr": hyperparameter_lr(root2), "decay": hyperparameter_decay(root2), "bs": hyperparameter_bs(root2), "epochs": hyperparameter_epochs(root2), "momentum" : hyperparameter_momentum(root2), "dropout": hyperparameter_dropout(root2) }
    print("The hyperparameter dictionary:", hyperparameter_dict)

    check_dictionary( hyperparameter_dict )

    # hp won't be accessible if it is not synthesized within the called function (where data input happens) i.e. within collect_parameters
    print("The hyperparameter array:", hp)


    # Collecting all hyperparameter values from the dictionary
    lr = hyperparameter_dict['lr']
    decay = hyperparameter_dict['decay']
    bs = hyperparameter_dict['bs']
    epochs = hyperparameter_dict['epochs']
    momentum = hyperparameter_dict['momentum']
    dropout = hyperparameter_dict['dropout']


    hyper_final = [lr, decay, bs, epochs, momentum, dropout]

    # Calling show_x() again generating the widget again: root2.destroy() to suppress this widget generation
    param_final = show_x()
    root2.destroy()

    print ("The final parameter to be trained:",  param_final)

    print("The final run hyperparameters in an array: ", hyper_final)


    check = parameter_selection()
    check_model = model_selection()


    print("The check: ", check)
    print("The check_model: ", check_model)

    train_status = 0 

    

    if check == "inhibition constant (Ki)" and check_model == "CNN & MLP hybrid model":

        # status, scaling_factor = diff_train(lr, decay, bs, epochs, momentum_val, dropout_val, "./dataset")
        # status, scaling_factor_diff = diff_train(lr, decay, bs, epochs, momentum, dropout, "./dataset")
        status, scaling_factor_diff = CNN_MLP_train_test(lr, decay, bs, epochs, momentum, dropout, "./dataset")
        print("The scaling factor in data: ", scaling_factor_diff)        
  


    if status != "": # This means that training executed and model made

        # messagebox.showinfo(title = 'Training status', message = "Training completed....proceed to analysis")
        messagebox.showinfo(title = 'Training status', message = "Training completed, check results in 'output_results' folder. We are working on the query, screen and analyze features of this software. We will update soon.")
        train_status = 1    

    root2.quit


    # This analysis is different from analyze_direct_query(scaling) function as analyze() simply runs an analysis/ query after training only as the scaling_factor is needed as input 
    #analyze_direct_query(scaling) runs the query directly, w/o training, based on existing model   

    def analyze():

        # image_path = str(open_image_file())

        # print(image_path)

        # print(image_path.rsplit('/')[-1])

        # input_image = image_path.rsplit('/')[-1]


        if check == "inhibition constant (Ki)" and check_model == "CNN & MLP hybrid model":

            # predicted_val = diff_query(input_image, scaling_factor_diff).flatten()

            # messagebox.showinfo(title = 'Query status', message = "Query completed. The predicted value is {}".format(predicted_val))
            
            messagebox.showinfo(title = 'Query status', message = "Analyze feature will be enabled soon")



    # Invoking the analysis (query) after training
    if  (train_status == 1 and check_model == "CNN & MLP hybrid model") : # train_status becomes 1 after training and model made

        b0 = tk.Button(root, text='Analyze',command = analyze)
        b0.grid(column = 3, row = 5)

    # return scaling_factor   


# #...........................................................................................................................................................................................................................................................................................................................................................



# #........................................................ This is where hyperparameters are taken as input and analysis is run ............................................................................................................................................-------------------------------------------------------------...................

# Defining the parameter input string
drop_label_return = StringVar()

# Defining the parameter & hyperparameter input function
def show_x():

    # IMPORTANT: If you don't declare lr, decay etc. as global, you cannot access them outside this function
    global lr, decay, bs, epochs, momentum, dropout
    global drop_label_return
    global drop_label_return2

    global hp 
    hp = []

    # # # instantiate an empty parameter dictionary: We declerate it here, since the data flow starts from this function
    global hyperparameter_dict
    hyperparameter_dict = {}


    drop_label = Label(root, text = click.get() + ' has been activated for prediction')

    if click.get() == "inhibition constant (Ki)":
        # print("Selected parameter", click.get())
        drop_label_return = click.get()
        print("Selected variable/parameter:", drop_label_return)

    # Basically showing the result of a click (on OptionMenu)
    drop_label.grid(column = 3, row =3)


##----------------------------------------------------------------------------------------------------------------------------------------------

    # This function directly runs the query w/o going into the training
    def analyze_direct_query():

        # image_path = str(open_image_file())

        # print(image_path)

        # print(image_path.rsplit('/')[-1])

        # input_image = image_path.rsplit('/')[-1]



        if check == "inhibition constant (Ki)" and check_model == "CNN & MLP hybrid model":

            # predicted_val = diff_query(input_image, scaling_factor_diff).flatten()

            # messagebox.showinfo(title = 'Query status', message = "Query completed. The predicted value is {}".format(predicted_val))
            
            messagebox.showinfo(title = 'Query status', message = "Query feature will be enabled soon")


        

    # Invoking check_model to use in if statements below        
    model_selection()

    if (check_model == "CNN & MLP hybrid model"):      
        skip_btn_query = tk.Button(root,text="Skip training and use an existing model to query",command = lambda: (analyze_direct_query()) )
        skip_btn_query.grid(column = 1, row =5)



    # def open_multiple_image_file():

    #     filepaths = filedialog.askopenfilename(parent = root, multiple = True, initialdir = os.getcwd(),title = "select a file and double click on it to import", filetypes = [("img files", "*.png")])    

    #     img_num = len(filepaths)


    #     input_image = ["" for i in range(img_num)] # declering empty string array in python

    #     input_list = list()

    #     for i in range(0,img_num):


    #         input_filepath = filepaths[i].rsplit('/')[-1]

    #         print(input_filepath)

    #         input_image[i] = input_filepath

    #         input_list.append(input_image[i])

    #     print("Input images array", input_image)   
    #     print("Input images list",  input_list)   

    #     return  input_list


    # This function directly runs the screening w/o going into the training 
    def analyze_direct_screen():

        # image_paths = open_multiple_image_file()

        # print("The current drop_label_retun: ",drop_label_return )

        # Invoking check_model to use in if statements below        
        model_selection()


        if drop_label_return == "inhibition constant (Ki)" and check_model == "CNN & MLP hybrid model":

            # predicted_val_screen = diff_screen(image_paths, scaling_factor_diff).flatten() 
            # messagebox.showinfo(title = 'Query status', message = "Screening completed. The predicted values are {}".format(predicted_val_screen))  

            messagebox.showinfo(title = 'Query status', message = "Screen feature will be enabled soon")

    def printToConsole(text_area):

        global SMILES_array
      
        SMILES_text = text_area.get("1.0 ","end-1c")

        # print(SMILES_text.split("\n"))

        SMILES_array = SMILES_text.split("\n")

        for i in range (0, len(SMILES_array)):

            if SMILES_array[i] == '': # only work if the cursor stays in the next to last line, not after the next to last line 

                SMILES_array.remove(SMILES_array[i])

        print(SMILES_array)

        return SMILES_array


    if (check_model == "CNN & MLP hybrid model") :      
        skip_btn_query = tk.Button(root,text="Skip training and use an existing model to screen",command = lambda: (analyze_direct_screen()) )
        skip_btn_query.grid(column = 4, row =5)




    # Under show_x() function, if buttons are pressed and values are given, it collects the values to construct the hyperparameter dictionary and training run        
    if click.get() != "":

        # Initiate the widget browser
        root2 = tk.Tk()
        root2.title('Define Hyperparameters')
        # root2.geometry("400x400")

        # #Define the canvas window for the widget

        widget = tk.Canvas(root2, width = 400, height = 400)
        widget.grid(columnspan = 4, rowspan = 9)

        hyperparameter_label = Label(root2, text = "Enter the hyperparameter values")
        hyperparameter_label.grid(column = 0, row =0)

        # hyperparameter value input
        # ---------------------------------------------------------------------------
        lr_label = Label(root2, text = "Enter the learning rate (lr)")
        lr_label.grid(column = 0, row =1)

        lr = Entry(root2)
        lr.insert(0, "0")
        lr.grid(column = 1, row =1)

        decay_label = Label(root2, text = "Enter the decay")
        decay_label.grid(column = 0, row =2)

        decay = Entry(root2 )
        decay.insert(0, "0")
        decay.grid(column = 1, row =2)

        batch_label = Label(root2, text = "Enter the batch size")
        batch_label.grid(column = 0, row =3)

        bs= Entry(root2)
        bs.insert(0, "0")
        bs.grid(column = 1, row =3)

        epochs_label = Label(root2, text = "Enter # of epochs")
        epochs_label.grid(column = 0, row =4)

        epochs = Entry(root2)
        epochs.insert(0, "0")
        epochs.grid(column = 1, row =4)

        momentum_label = Label(root2, text = "Enter momentum (avoid for adam opt) ")
        momentum_label.grid(column = 0, row = 5)

        momentum = Entry(root2)
        momentum.insert(0, "0")
        momentum.grid(column = 1, row = 5)


        dropout_label = Label(root2, text = "Enter dropout (avoid for CNNs) ")
        dropout_label.grid(column = 0, row =6)

        dropout = Entry(root2)
        dropout.insert(0, "0")
        dropout.grid(column = 1, row = 6)



        b1 = tk.Button(root2, text='Confirm lr',command= lambda: hyperparameter_lr(root2))
        b1.grid(column = 2, row =1)

        b2 = tk.Button(root2, text='Confirm decay',command= lambda : hyperparameter_decay(root2))
        b2.grid(column = 2, row =2)

        b3 = tk.Button(root2, text='Confirm batch',command= lambda: hyperparameter_bs(root2))
        b3.grid(column = 2, row =3)

        b4 = tk.Button(root2, text='Confirm epochs',command= lambda: hyperparameter_epochs(root2))
        b4.grid(column = 2, row =4)

        b5 = tk.Button(root2, text='Confirm momentum',command= lambda: hyperparameter_momentum(root2))
        b5.grid(column = 2, row =5)

        b6 = tk.Button(root2, text='Confirm dropout',command= lambda: hyperparameter_dropout(root2))
        b6.grid(column = 2, row =6)

        b7 = tk.Button(root2, text='Run',command = lambda: collect_parameters(root2, root))
        b7.grid(column = 2, row =7)

        b8 = tk.Button(root2, text='Exit',command= root2.destroy)
        b8.grid(column = 2, row =8)
 
        
        b9 = tk.Button(root2,text="Skip training and Analyze",command = lambda: analyze_direct_query())
        b9.grid(column = 0, row =8)

        root2.quit



    return drop_label_return   

# Choose the parameter to be modeled
x = ["inhibition constant (Ki)", ""]         

click = tk.StringVar()
click.set(x[1])

drop_label = Label(root, text = "Training variable/parameter")
drop_label.grid(column = 3, row =0)

drop = OptionMenu(root,click,*x)
drop.grid(column = 3, row =1)


parameter_choice = tk.StringVar()
# drop_btn = tk.Button(root,text="Activate the parameter selection",command = show_x )
drop_btn = tk.Button(root,text="Activate the parameter selection",command = lambda: [ show_x() , parameter_selection() ])
drop_btn.grid(column = 3, row =2)
parameter_choice.set("Select the property parameter to predict")




# Choose the model after parameter selection
xx = ["CNN & MLP hybrid model",""]         

click2 = tk.StringVar()
click2.set(xx[1])

drop_label2 = Label(root, text = "Training model")
drop_label2.grid(column = 4, row =0)

drop2 = OptionMenu(root,click2,*xx)
drop2.grid(column = 4, row =1)


parameter_choice2 = tk.StringVar()
drop_btn2 = tk.Button(root,text="Activate the model selection",command = lambda: [ show_x() , model_selection() ])
drop_btn2.grid(column = 4, row =2)
parameter_choice2.set("Select the model to train the property")



# Start the application
root.mainloop()




