# Code for Thesis Presentation
This is a Thesis Presentation code provided by Nguyen Vinh Tri, and here is how to use the code:
## 1. Setup
### pip
In case you are using pip, please run: 

```pip install -r requirement.txt```

### conda
In case you are using conda, please run:

``` conda install --file requirement.txt ```
## 2. Get the classification model
Please get the classification model by following this link: [Drive](https://drive.google.com/drive/folders/1ytZLckHCX0BNCf6YhrdKVjTxxwQT4Zsl?usp=sharing)

Please ensure the models are placed in the ***thesis_model*** folder. Ex: *thesis_model/model_name.pth*
## 3. Run code
After setup and downloading all the models, the code will be run by calling: ```streamlit run demo.py```

# UI usage
First, it is necessary to select the model in the left panel. The default model is the phobert_lstm model. 

Next, please input the data. The sample data will be placed in the ***data*** folder. Please follow the format of the sample file by appending the new rows to the sample file. 

There is another way to input the data. The left panel contains the text area for inputting texts. The model also classifies the text input in the text area. 

Finally, the results will be displayed in 2 tables. The first table illustrates the probabilities of 7 emotions for each row. The second table illustrates the label of each row
# Note:
The report and the video are stored in this [link](https://drive.google.com/drive/folders/1GucSh8c4J6cgqT-683ICTOHJr9AxxHjp?usp=sharing)
