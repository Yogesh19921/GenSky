# GenSky

## Genetic Anomaly Detector
Gensky is based on Linear Multi Objective Genetic Programming. Genetic programming is a field of Computer Science which involves developing self evolving programs. Multi Objective Genetic Programming, hence, is a field to develop self evolving programs which are suited to achieve multiple objectives. Here, we have two objectives i) detecting attacks and ii) detecting normal traffic. Both the objectives are equally important and Multi Objective GP considers them both equally important.

### How to install and test

       1. Install pip.
       2. Using pip, install scikit-learn, numpy, pandas, and matplotlib.
       3. Open the current directory in terminal.
       4. Run python main.py. You should get something like this. 
       Here 6349 is the number of instances of class 1 and 2670 of class 2 in the train set. More in wiki
   ![picture alt](http://i.imgur.com/dL1WQQG.png)
       
   
### PreProcessing your dataset
If possible, i will provide with a script to pre process the dataset. Until then,

       1. Get rid of unnecessary features from your dataset like flow or source port, destination port etc.
       2. Convert every feature to numerical fields. Eg. tcp-1 , udp-2, Anomalies- 1, normal traffic -2. 
          Use your own imagination. Refer to already provided dataset for hints.
          
 
### Running the Genetic Program on a dataset
       1. After you have preprocessed, put all the datasets in the datasets folder.
       2. The program is multithreaded, so it will simultaneously process all the datasets.
          It might take long, if the dataset is huge like ISOT dataset.
       3. Find the results in the output folder with corresponding file names.
          Result looks something like this.
          Here in Train, 95% accuracy is for detecting attacks and 70.5% for detecting 
          normal traffic.
          Similarly in test, 96.19% accuracy is for detecting attacks and 
          64.72% for detecting normal traffic.
 ![picture alt](http://i.imgur.com/vebo1Qv.png )   
 
 ### Architecture and working of GP
 Refer to the [wiki](https://github.com/Yogesh19921/GenSky/wiki)
