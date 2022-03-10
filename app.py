import pandas as pd

import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin

#Practice.
# df=pd.read_csv('https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv')
# print(df.head())
# pf=ProfileReport(df)
# pf.to_file('report.html')
# x=df[['TV']]
# y=df['Sales']
# lm=LinearRegression()
# lm.fit(x,y)
# lm.score(x,y)
# print(lm.score(x,y))

app = Flask(__name__)
@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homepage():
    return render_template("index.html")

@app.route('/report',methods=['POST','GET'])  # route to display the home page
@cross_origin()
def analysisPage():
    return render_template("report.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            gre_score=float(request.form['GRE'])
            toefl_score = float(request.form['toeffel'])
            university_rating = float(request.form['University'])
            sop = float(request.form['SOP'])
            lor = float(request.form['LOR'])
            cgpa = float(request.form['CGPA'])
            is_research = request.form['Research']
            if(is_research=='yes'):
                research=1
            else:
                research=0
            filename = 'Admission_lr_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction=loaded_model.predict([[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]])
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('result.html',prediction=round(10*prediction[0]))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app

# class model:
#     def __init__(self,df):
#         self.df=df
#
#     def head_display(self):
#         return(df.head())
#
#     def check_df_info(self):
#         return (df.info())
#
#     def profile_report(self):
#         pf=ProfileReport(df)
#         pf.to_notebook_iframe()
#         pf.to_file('report1.html')
