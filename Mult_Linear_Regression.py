import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("C:\\Users\\ar358\\OneDrive\\Documents\\employee_salary_real_world_dataset.csv")
print(df)


#HANDEL MISSING DATA
df["Degree"]=df["Degree"].fillna("B.Tech")
df["City"]=df["City"].fillna("Jaipur")
df["Skill_Rating"].fillna(df["Skill_Rating"].mean(),inplace=True)
df["Projects_Completed"].fillna(df["Projects_Completed"].mean(),inplace=True)
print(df["City"].unique())
print(df.isnull().sum())

#============================
#LABEL ENCODING
#============================

degree_encoder = LabelEncoder()
company_encoder = LabelEncoder()
city_encoder = LabelEncoder()

df["Degree"]=degree_encoder.fit_transform(df["Degree"])
df["Company_Size"]=company_encoder.fit_transform(df["Company_Size"])
df["City"]=city_encoder.fit_transform(df["City"])

X=df[["Age","Degree","Passing_Year","Experience_Years","Company_Size","City","Skill_Rating","Projects_Completed"]]
Y=df["Salary"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#============================
#MODEL EVOULATION
#============================

mae=mean_absolute_error(Y_test,Y_pred)
mse=mean_squared_error(Y_test,Y_pred)
r2=r2_score(Y_test,Y_pred)
rmse=np.sqrt(mse)

print("-------------MODEL EVOLULATION-------------")
print("MAE=",round(mae,2))
print("MSE=",round(mse,2))
print("R2 SCORE:",round(r2,2))
print("RMSE=",round(rmse,4))

#============================
#TRAIN TEST R2 SCORE
#Here we check how model work on ttrain data and unseen data.
#============================
print("TRAIN R2:",r2_score(Y_train,model.predict(X_train)))
print("TEST R2:",r2_score(Y_test,model.predict(X_test)))

#============================
#DATA VISUALIZATION
#============================

#AGE VS SALARY(SCATTER PLOT)
plt.subplot(3,2,1)
plt.scatter(df["Age"],df["Salary"],marker="d")
plt.xlabel("EMPLOYEE AGE")
plt.ylabel("EMPLOYEE SALARY")
plt.title("AGE VS SALARY")
plt.grid(True)

#EXPERIANCE VS SALARY(BAR PLOT)
plt.subplot(3,2,2)
plt.bar(df["Experience_Years"],df["Salary"])
plt.xlabel("EXPERIANCE")
plt.ylabel("SALARY")
plt.title("EXPERIANCE VS SALARY")
plt.grid(True)

#ACTUAL VS PREDICTED SALARY(HISTOGRAM)
plt.subplot(3,2,3)
plt.scatter(Y_test,Y_pred)
plt.xlabel("ACTUAL")
plt.ylabel("PREDICTED SALARY")
plt.title("ACTUAL VS PREDICTED SALARY")
plt.grid(True)

#RESIDUAL PLOT (IT IS USE TO FIND ERROR(RESIDUAL= ACTUAL SALARY - PREDICTTED SALARY))
#SCATTER PLOT
plt.subplot(3,2,4)
residual=Y_test - Y_pred
plt.scatter(Y_test,residual)
plt.axhline(y=0,color="red")
plt.xlabel("PREDICTED SALARY")
plt.ylabel("RESIDUAL ERROR")
plt.title("RESIDUAL PLOT ")
plt.grid(True)

#DEGREE PIE CHART
degree_name=degree_encoder.inverse_transform(df["Degree"])
degree_count=pd.Series(degree_name).value_counts()

plt.subplot(3,2,5)
plt.pie(degree_count,labels=degree_count.index,autopct="%1.1f%%")
plt.title("DEGREE DISTRIBUTION")
plt.legend(loc="center left",bbox_to_anchor=(2,0.5))

#CITY HISTOGRAM
city_name=city_encoder.inverse_transform(df["City"])
plt.subplot(3,2,6)
plt.hist(city_name,edgecolor="black")
plt.title("CITY DISTRIBUTION")
plt.xlabel("CITY")
plt.ylabel("FREQUENCY")
plt.xticks(rotation=45)



plt.tight_layout()
plt.show()


#============================
#USER INPUT
#============================

print("ENTER YOUR DETAILS")
age=int(input("ENTER YOUR AGE:"))

degrees=degree_encoder.classes_
while True:
    print("\nAVAILABLE DEGREE:")
    for i,d in enumerate(degrees,1):
        print(f"{i}.{d}")
    try:
            choice=int(input("CHOOSE DEGREE:"))
            if 1 <= choice <= len(degrees):
                degree=degrees[choice - 1]
                print("SELECTED DEGREE:",degree)
                break
            else:
                print("WRONG CHOICE! PLEASE SELECT AGAIN:")
    except ValueError:
            print("PLEASE ENTER NUMBER ONLY:")

passing_year=int(input("ENTER PASSING YEAR:"))
experience_years=int(input("ENTER YOUR EXPERIENCE:"))

company_size=company_encoder.classes_
while True:
    print("AVILABLE SIZE:")
    for i,s in enumerate(company_size,1):
        print(f"{i}.{s}")
    try:
            choice=int(input("ENTER SIZE:"))
            if 1 <= choice <= len(company_size):
                size=company_size[choice - 1]
                print("SELECTED SIZE:",size)
                break
            else:
                print("WRONG CHOICE! PLEASE SELECT AGAIN")
    except ValueError:
            print("PLEASE ENTER NUMBER ONLY")

    


cities=city_encoder.classes_
while True:
    print("\nAVAILABLE CITY:")
    for i,c in enumerate(cities,1):
        print(f"{i}.{c}")
    try:
            choice=int(input("CHOOSE CITY:"))
            if 1 <= choice <= len(cities):
                city=cities[choice - 1]
                print("SELECTED CITY:",city)
                break
            else:
                print("WRONG CHOICE PLEASE SELECT AGAIN:")
    except ValueError:
            print("PLEASE ENTER NUMBER ONLY:")

skill_rating=float(input("RATE YOUR SKILL FROM(1-10):"))
project_completed=int(input("ENTER NUMBER OF PROJECT YOU COMPLETED:"))

degree=degree_encoder.transform([degree])[0]
size=company_encoder.transform([size])[0]
city=city_encoder.transform([city])[0]

New_Data=pd.DataFrame({"Age":[age],
                       "Degree":[degree],
                       "Passing_Year":[passing_year],
                       "Experience_Years":[experience_years],
                       "Company_Size":[size],
                       "City":[city],
                       "Skill_Rating":[skill_rating],
                       "Projects_Completed":[project_completed]})

salary=model.predict(New_Data)
print("YOUR EXPECTED SALARY IS AROUND=",round(salary[0],2))

