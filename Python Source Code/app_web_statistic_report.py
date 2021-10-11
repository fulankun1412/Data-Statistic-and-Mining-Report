from flask import Flask, render_template
import mysql.connector
import numpy as np
from numpy.core import records
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from wordcloud import WordCloud
from sklearn.preprocessing import PolynomialFeatures
import os

pd.set_option('display.float_format', lambda x: '%.2f' % x)

app = Flask(__name__)

def connect_MySql_and_fetch(table_Name):
    connection = mysql.connector.connect(host='localhost', 
                                        database='integrated_data', 
                                        user='lanang_afkaar', 
                                        password='1q2w3e4r5t')
    sql_select_Query = "SELECT * FROM {}".format(table_Name)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    num_records = cursor.rowcount
    print("Total number of rows in table: ", num_records)
    
    return records, num_records

def visualize_data_employee_statistic(maritals, departments, genders, productivities):
    # Count How many Marital status in the company
    Single = 0
    Married = 0
    Divorced = 0
    Separated = 0
    Widowed = 0

    for status_id in maritals:
        if status_id == 0 :
            Single +=1
        elif status_id == 1:
            Married +=1
        elif status_id == 2:
            Divorced +=1
        elif status_id == 3:
            Separated +=1
        else:
            Widowed +=1
    
    plot_Marital = [Single, Married, Divorced, Separated, Widowed]
    labels_Marital = ['Single', 'Married', 'Divorced', 'Separated', 'Widowed']

    # Count How many Each's Employee Department
    AdminOffices = 0
    ExecutiveOffice = 0
    IT_IS = 0
    SoftwareEngineering = 0
    Production = 0
    Sales = 0

    for department_id in departments:
        if department_id == 1 :
            AdminOffices +=1
        elif department_id == 2 :
            ExecutiveOffice +=1
        elif department_id == 3 :
            IT_IS +=1
        elif department_id == 4 :
            SoftwareEngineering +=1
        elif department_id == 5 :
            Production +=1
        else:
            Sales +=1

    plot_department = [AdminOffices, ExecutiveOffice, IT_IS, SoftwareEngineering, Production, Sales]
    labels_department = ['Admin Offices', 'Executive Office', 'IT/IS', 'Software Engineering', 'Production', 'Sales']

    # Count how diverse is the gender in the company
    Male = 0
    Female = 0
    for gender in genders:
        if gender == "F":
            Female += 1
        else:
            Male += 1

    plot_gender = [Male, Female]
    labels_gender = ['Male', 'Female']

    # Count How many percentage effective employee
    effective = 0
    not_effective = 0

    for productivity in productivities:
        if productivity >= 1.0:
            effective += 1
        else: 
            not_effective += 1

    plot_productivity = [effective, not_effective]
    label_productivity = ['100% or More','Less than 100%']

    # Plot the Pie Chart for each graph
    img_marital_report = "static/marital_report.png"
    img_department_report = "static/department_report.png"
    img_gender_report = "static/gender_report.png"
    img_productivity_report = "static/productivity_report.png"

    fig, ax = plt.subplots()
    ax.pie(plot_Marital, labels = labels_Marital, autopct='%.1f%%')
    ax.set_title('Marital Status')
    if os.path.exists(img_marital_report) == True:
        os.remove(img_marital_report)
    plt.savefig(img_marital_report)

    fig, ax = plt.subplots()
    ax.pie(plot_department, labels = labels_department, autopct='%.1f%%')
    ax.set_title('Department Distribution')
    if os.path.exists(img_department_report) == True:
        os.remove(img_department_report)
    plt.savefig(img_department_report)

    fig, ax = plt.subplots()
    ax.pie(plot_gender, labels = labels_gender, autopct='%.1f%%')
    ax.set_title('Gender Diversity')
    if os.path.exists(img_gender_report) == True:
        os.remove(img_gender_report)
    plt.savefig(img_gender_report)

    fig, ax = plt.subplots()
    ax.pie(plot_productivity, labels = label_productivity, autopct='%.1f%%')
    ax.set_title('Productivity Effectiveness')
    if os.path.exists(img_productivity_report) == True:
        os.remove(img_productivity_report)
    plt.savefig(img_productivity_report)

def visualize_data_houses_statistic(grade, yr_built, price):
    img_grade_report = "static/grade_report.png"
    img_built_count_report = "static/built_count_report.png"
    img_price_dist_report = "static/price_dist_report.png"
    img_most_built_year_report = "static/built_year_report.png"

    # Visualize count how many each grade put them into histogram
    x = grade
    plt.subplots(sharey=True, tight_layout=True)
    bins = 12
    # Freedman–Diaconis Rule to count appropriate Bins *Optional
    #q25, q75 = np.percentile(x, [0.25, 0.75])
    #bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    #bins = int(round((x.max() - x.min()) / bin_width))
    plt.hist(x, bins=bins)
    plt.title('Grade Counts')
    plt.ylabel("Counts")
    plt.xlabel("Grade")
    if os.path.exists(img_grade_report) == True:
        os.remove(img_grade_report)
    plt.savefig(img_grade_report)

    # Visualize count how many each years house built put them into histogram
    x = yr_built
    plt.subplots(sharey=True, tight_layout=True)
    bins = 100
    # Freedman–Diaconis Rule to count appropriate Bins *Optional
    #q25, q75 = np.percentile(x, [0.25, 0.75])
    #bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    #bins = int(round((x.max() - x.min()) / bin_width))
    plt.hist(x, bins=bins)
    plt.title('Built Year Counts')
    plt.ylabel("Built Counts")
    plt.xlabel("Years")
    if os.path.exists(img_built_count_report) == True:
        os.remove(img_built_count_report)
    plt.savefig(img_built_count_report)

    # Visualize count how many each price and the distribution
    x = price
    plt.subplots()
    plt.boxplot(x, vert=True)
    plt.title('Price Distribution')
    plt.xlabel('House')
    plt.ylabel('Price in Million')
    if os.path.exists(img_price_dist_report) == True:
        os.remove(img_price_dist_report)
    plt.savefig(img_price_dist_report)

    # Visualize and find out which year on most houses built
    builtyear = pd.DataFrame({"Years":yr_built})
    builtyear["Years"] = builtyear["Years"].apply(lambda x: "y" + str(x)) #I can't use wordcloud with integers so I put y on head
    builtyear["Years"].head()

    plt.subplots(figsize=(5,5))
    wcloud  = WordCloud(background_color="white",width=250,height=250).generate(",".join(builtyear["Years"]))
    plt.imshow(wcloud)
    plt.title("Years for Most Built Homes",fontsize=20)
    plt.axis("off")
    if os.path.exists(img_most_built_year_report) == True:
        os.remove(img_most_built_year_report)
    plt.savefig(img_most_built_year_report)

def linear_regression_prediction(y, x):
    x_array = np.array(x)
    y_array = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.2, random_state=0)
    
    LR = LinearRegression()
    LR.fit(x_train, y_train)
    y_pred = LR.predict(x_test)

    df_coeff = pd.DataFrame(LR.coef_, x.columns, columns=['Coefficient'])
    df_coeff = df_coeff.apply(np.floor)
    df_prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_prediction = df_prediction.apply(np.floor)

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    df_score = pd.DataFrame([mae, mse, rmse], ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'], columns=['Score'])
    df_score = df_score.apply(np.floor)
    return df_prediction, df_coeff, df_score

@app.route("/")
def home():
    # Connect, Describe, Visualize, Create Prediction for table "salary_and_employee_perfomances_2020"
    records_employee, num_employee_records = connect_MySql_and_fetch(table_Name="salary_and_employee_perfomances_2020")

    #
    df_employee = pd.DataFrame(list(records_employee), columns=['Name', 'Employee_ID', 
                                                    'Marital_ID', 'MaritalStatus_ID', 'Marital_Description', 
                                                    'Gender_ID', 'Gender', 
                                                    'Department_ID', 'Department', 
                                                    'Position_ID', 'Position',
                                                    'Productivity_Rate_in_a_year', 'Salary'])
    
    #
    df_employee['Productivity_Rate_in_a_year'] = pd.to_numeric(df_employee['Productivity_Rate_in_a_year'],errors = 'coerce')
    df_employee_described = df_employee.describe()

    # 
    visualize_data_employee_statistic(df_employee['MaritalStatus_ID'], df_employee['Department_ID'], df_employee['Gender'],df_employee['Productivity_Rate_in_a_year'])
    
    #
    df_predictSalary = df_employee.drop(['Name', 'Employee_ID', 'Marital_Description', 'Gender', 'Department', 'Position'], axis = 1)
    y = df_predictSalary.pop("Salary")
    x = df_predictSalary.copy()
    predict_salary, predict_salaryCoeff, predict_salaryScore = linear_regression_prediction(y, x)

    # Connect, Describe, Visualize, Create Prediction for table "prices_estimation_for_house_sales"
    records_houses, num_houses_records = connect_MySql_and_fetch(table_Name="house_sales_records")

    #
    df_houses = pd.DataFrame(list(records_houses), columns=['price', 
                                                   'bedrooms','bathrooms',
                                                   'sqft_living','sqft_lot','floors',
                                                   'waterfront','view','condition','grade',
                                                   'sqft_above','sqft_basement','yr_built','yr_renovated',
                                                   'zipcode','lat','long','sqft_living15','sqft_lot15'])
    
    #
    df_houses['price'] = pd.to_numeric(df_houses['price'],errors = 'coerce')
    df_houses['bathrooms'] = pd.to_numeric(df_houses['bathrooms'],errors = 'coerce')
    df_houses['floors'] = pd.to_numeric(df_houses['floors'],errors = 'coerce')
    df_houses['lat'] = pd.to_numeric(df_houses['lat'],errors = 'coerce')
    df_houses['long'] = pd.to_numeric(df_houses['long'],errors = 'coerce')
    df_houses_described = df_houses.describe()
    
    #
    visualize_data_houses_statistic(df_houses.grade, df_houses.yr_built, df_houses.price)
    
    #
    y = df_houses.pop("price")
    x = df_houses.copy()
    predict_price, predict_priceCoeff, predict_priceScore = linear_regression_prediction(y, x)

    #
    return render_template("index.html",
                         tables_employee = [df_employee_described.to_html(classes = 'data')], 
                         tables_salaryPrediction = [predict_salary.to_html(classes = 'data')],
                         tables_salaryCoeff = [predict_salaryCoeff.to_html(classes = 'data')],
                         tables_salaryScore = [predict_salaryScore.to_html(classes = 'data')],
                         rmse_salaryScore = predict_salaryScore['Score']['Root Mean Squared Error'],
                         num_employee = num_employee_records,
                         tables_houses = [df_houses_described.to_html(classes = 'data')],
                         tables_pricePrediction = [predict_price.to_html(classes = 'data')],
                         tables_priceCoeff = [predict_priceCoeff.to_html(classes = 'data')],
                         tables_priceScore = [predict_priceScore.to_html(classes = 'data')],
                         rmse_priceScore = predict_priceScore['Score']['Root Mean Squared Error'],
                         num_houses = num_houses_records
                         )


if __name__ == "__main__":
    app.run()