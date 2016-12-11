# Our code was contain in 6 seperate file:

1. Final Report-12-11.ipynb
    Contains some illustration of 
2. demographic_data.py
    Contains helper function to extract the demographic data.
3. weather.py
    Contains helper function to extract the weather data.
4. kde.py
    Contains KDE helper function.
5. KDE_pre_computation.py
    Contains pre compute KDE helper function and crime counts helper function.
    Contains main function that contains whole workflow of precompute KDE.
6. KDE_visualization.py
    Contains function to visualize KDE.
    Contains main function that can show KDE visualisation.


==============================
# How to run our code:

Note: Make sure you have all of the data folder in the same directory,
      All the files are in our google drive: 
      https://drive.google.com/drive/folders/0ByHpjY5qy8GpeVFySzBLd3RZMWc?usp=sharing
	1. ghcn-daily
	2. demographic_data
	3. data

1.1 Parsing Weather Data
	Run the weather.py file in the terminal directly, in the same directory 
    	of extracted_data folder and ghcn-daily foler.
1.2 Use Weather Data API
    	This is illustrated in the notebook 
2.1 Parsing demographic Data
    	To parse data and generate pickle file, run the weather.py script in the terminal,
    	in the same directory of demographic_data folder.
    	(This is also illustrated in the notebook)
2.2 Use Weather Data API
    	This is illustrated in the notebook 
3.1 Parsing Crime Data 
    	Run KDE_pre_computation.py script in the terminal,
    	in the same directory of data folder.
3.2 Visualize KDE
    	Run KDE_visualization.py script in the terminal, in the same directory of data folder.
    	If you don’t have ‘data/chicago_crime_data_2015_sorted.csv’ file yet, 
        run KDE_pre_computation.py to generate this file first. 
4.1 Model Training
	model.py contains main function to run the model training.
	The code of model training is in the notebook. 
	You can run code in the notebook directly.
