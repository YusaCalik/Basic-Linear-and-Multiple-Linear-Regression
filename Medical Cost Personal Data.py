import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
df=pd.read_csv('insurance.csv')
df.head(5)

#Analyze Data
df.info()
df.describe().T


df.isnull().values.any() #There are no missing observations in the data
df.corr()#we are looking at correlation

#we study correlation broadly
import seaborn as sns
sns.pairplot(data=df,kind='reg')
sns.jointplot(x='age',y='charges',data=df)
sns.jointplot(x='age',y='charges',data=df,kind='reg')


from sklearn.linear_model import LinearRegression 
import statsmodels.formula.api as smf
# I considered data above 10k as outliers and therefore excluded it from the data.
below_10000=df[df['charges']<10000]
X=below_10000[["age"]]
y=below_10000[["charges"]]
X
y
reg=LinearRegression()
model=reg.fit(X,y)

#I took the first 10 data of the charges column and the first 10 data of the values I predicted 
#with the model I built and wanted to compare it with the naked eye
gercek_degerler=y[0:10]
gercek_degerler

model_predict=pd.DataFrame(model.predict(X)[0:10])
model_predict
k_t=np.concatenate([gercek_degerler, model_predict], axis=1)
k_t=pd.DataFrame(k_t)
k_t.columns =['gercek degerler', 'tahmin edilen degerler']
k_t
# -----------------------------------------------------------------

below_10000=df[df['charges']<10000]
below_10000
x_age=below_10000['age'].values
charges_below_10000=below_10000['charges'].values
x_age=x_age.reshape(-1,1)

#I set up linear regression between charge and age and got fit. Then I printed the results separately as graph and value.
x_train,x_test,y_train,y_test=train_test_split(x_age,charges_below_10000,test_size=0.20,train_size=0.80)

print('shape of x_train{0}'.format(x_train.shape))
print('shape of y_train{0}'.format(y_train.shape))
print('shape of x_test{0}'.format(x_test.shape))
print('shape of y_test{0}'.format(y_test.shape))
x_test
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
plt.xlabel('Ages')
plt.ylabel('Charges')
plt.title('Training Data')

plt.scatter(x_test,y_test)
plt.xlabel('Ages')
plt.ylabel('Charges')
plt.title('Testing Data')

reg.fit(x_train, y_train)
charge_prediction = reg.predict(x_test)
print(f"Test Score: {reg.score(x_test,y_test)*100}%")
print(f"Train Score: {reg.score(x_train,y_train)*100}%")

plt.scatter(x_train, y_train, color="orange")
plt.plot(x_test, charge_prediction)
plt.xlabel("Age")
plt.ylabel("Charges in $")
plt.title("Linear Regression")
plt.plot
# =============================================================================================================================================
# 1.I noticed that the data looks linear on two different levels. The first is linear relationship is for Age vs. Charge under ten thousand dollars". This most likely means that if you do not have an expensive illness, as you age you're Charge will increase.
# 2.If you have a very expensive illness (above ten thousand dollars), then it is also linear with a highter y-intercept value or Charge
# 3.Some people with a lower age have a higher Charge than people who are older, so age is not the only metric which charge is based.
# ==================================================================================================================================

# =============================================================================
# Multiple Linear Regression 
# =============================================================================
from sklearn.preprocessing import LabelEncoder # import LabelEncoding
data=pd.read_csv('insurance.csv')
data.head()
le = LabelEncoder()
#3 different categorical columns is changed numeric columns to take them for model.
categorical_columns = ['sex', 'smoker', 'region']
df_encoded = pd.get_dummies(data = data, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first = True,
              dtype='int8')
df_encoded.columns
df_encoded

x=df_encoded.drop(['charges'],axis=1)
y=data['charges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

print('shape of x_train{0}'.format(x_train.shape))
print('shape of y_train{0}'.format(y_train.shape))
print('shape of x_test{0}'.format(x_test.shape))
print('shape of y_test{0}'.format(y_test.shape))
reg=LinearRegression()
model=reg.fit(x_train,y_train)

# study to coefficient and intercept of model
model.coef_ 
model.intercept_

y_pred=model.predict(x_test)

model.score(x_test,y_test)
model.score(x_train,y_train)
# our score values are bad. 0.78 is not enough for prediction.
from sklearn.metrics import mean_squared_error
rmse= np.sqrt(mean_squared_error(y_train,model.predict(x_train)))
rmse

# =============================================================================
# HYPOTHESING TEST
# =============================================================================
#wrote a function that measures the significance of the p-value.
def htestcontrol(pvalue):
    significance_level=0.05
    if(pvalue<significance_level):
        print("Reject H0 hypothesis.Null Hypothesis is rejected.")
    else:
        print("H0 is not rejected.Fail to reject Null Hypothesis.")
# =============================================================================
# Question 1 -
# To prove charges of people who do smoking are greater than those who don't.
# 
# Performing Right Tailed T-test
# Null Hypothesis ( H0 ) - Charges are same for Smokers and Non-Smokers
# 
# Alternate Hypothesis ( HA ) - Charges are greater for smokers.
# 
# First Step -
# Checking for basic assumpitons for the hypothesis
# 
# Second step-
# Performing Right tailed t-test
# 
# Third step -¶
# Checking for hypothesis result
# 
# Checking for basic assumptions
# Normality check
# Homogeneity of Variances
# =============================================================================
# =============================================================================
# Here we consider a linear regression and Linear regression assumes that the errors 
# or residuals follow a normal distribution
# we can test with 3 ways.Q Q plot ile, Box plot ile, Histogram ile test edebiliriz.
# =============================================================================
# Checking Normality with First Way(Q Q or Quantile-Quantile Plot)
import pylab
import scipy.stats as stats
plt.figure(figsize=(10,6))
stats.probplot(df['charges'].apply(np.log),dist='norm',plot=plt)
plt.show()
# eğer doğal logaritmasını çizdirirsek normal dist oluyor. Diğer Yolları da deneyelim.

# Checking Normality with Second Way (Boxplot)
import seaborn as sns
sns.boxplot(df['charges'].apply(np.log))

pd.DataFrame({'normal':df.charges,"log":df['charges'].apply(np.log)})

sns.boxplot(df['charges'])

#  Checking Normality with Third Way (histogram,distplot)
sns.distplot(df.charges.apply(np.log))
df.head()

# Homogenous test and t-test
smokers=df[df['smoker']=="yes"]['charges'].apply(np.log)
non_smokers=df[df['smoker']=="no"]['charges'].apply(np.log)
from scipy.stats import levene
stat,p=levene(smokers,non_smokers)
print("p-value:",p)
htestcontrol(p)
#rejects the hypothesis. So this means that the variances are not equal.

st,p=stats.ttest_ind(smokers,non_smokers)
print("t-test p-value:",(p/2))
htestcontrol((p/2))
# =============================================================================
# To prove the BMI of females is different from that of males .
# 
# Performing Two Tailed T-test
# Null Hypothesis ( H0 ) - BMI for male and females are same.
# 
# Alternate Hypothesis ( HA ) - BMI for males and females are different.
# 
# First Step -
# Checking for basic assumpitons for the hypothesis
# 
# Second step-
# Performing Two tailed t-test
# 
# Third step -
# Checking for hypothesis result
# 
# Checking for basic assumptions
# Normality check
# Homogeneity of Variances
# =============================================================================
# Checking normal distribution 
sns.boxplot(df['bmi'].apply(np.log))
sns.distplot(df['bmi'].apply(np.log))
#we can say this two graphic explain us that the data is distributed normal

#Checking male and female homogenous for bmi
male_bmi=df[df['sex']=="male"]['bmi'].apply(np.log) 
female_bmi=df[df['sex']=="female"]['bmi'].apply(np.log) 

stat,p=levene(male_bmi,female_bmi)
print('homogenous p-value:',p)
htestcontrol(p)
#h0 is not rejected.So, variance of data is homogenous which means they are same.

#T-Test
stat,p=stats.ttest_ind(male_bmi,female_bmi)
print("t-test p-value:",p)
htestcontrol(p)
#This means bmi of male and female are same.

# =============================================================================
# To check if the proportion of smoking significantly different across different regions.
# 
# Performing Chi-Square test
# Null Hypothesis ( H0 ) - Proportion of smoking is equal across different regions.
# 
# Alternate Hypothesis ( HA ) - Proportion of smoking is different across different regions.
# 
# First step-
# Performing chi-square test
# 
# Second step -
# Checking for hypothesis result
# =============================================================================
# The chi-square test is applied to find out the existence of the relationship between two different categorical variables.
# Therefore, as a null hypothesis, we keep the positive side of the test, and in the alternative hypothesis, we keep the negative side.
data=pd.crosstab(df["region"],df["smoker"])
data.head()
probabilities=np.array(data.iloc[:,:])
chi2, p, dof, ex =stats.chi2_contingency(probabilities)
htestcontrol(p)
# We cannot reject the null hypothesis, which means that the smoking rate is the same in different regions. That is, the smoking rate is the same in different regions.

# =============================================================================
# To check if the mean BMI of women with 0 child , 1 child, and 2 children the same.
# 
# Performing One-way Anova
# Null Hypothesis ( H0 ) - Mean BMI for females of children 0,1,2 is same.
# 
# Alternate Hypothesis ( HA ) - Mean BMI for females of children 0,1,2 is different.
# 
# First Step -
# Checking for basic assumpitons for the hypothesis
# 
# Second step-
# Performing One-way Anova
# 
# Third step -
# Checking for hypothesis result
# 
# Checking for basic assumptions
# Normality check
# Homogeneity of Variances
# =============================================================================
# Checking normal distribution 
sns.boxplot(df['bmi'].apply(np.log))
sns.distplot(df['bmi'].apply(np.log))
#we can say this two graphic explain us that the data is distributed normal
df.head()
#Checking male and female homogenous for bmi
female = df[df['sex']=='female']
fem_ch_0 = female[female['children']==0]['bmi'].apply(np.log)
fem_ch_1 = female[female['children']==1]['bmi'].apply(np.log)
fem_ch_2 = female[female['children']==2]['bmi'].apply(np.log)
fem_ch_3 = female[female['children']==3]['bmi'].apply(np.log)
fem_ch_4 = female[female['children']==4]['bmi'].apply(np.log)
fem_ch_5 = female[female['children']==5]['bmi'].apply(np.log)


stat,p=levene(fem_ch_0,fem_ch_1,fem_ch_2,fem_ch_3,fem_ch_4,fem_ch_5)
print('homogenous p-value:',p)
htestcontrol(p)
#h0 is not rejected.So, variance of data is homogenous which means they are same.
# The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes.
from scipy.stats import f_oneway
stat,p=stats.f_oneway(fem_ch_0,fem_ch_1,fem_ch_2,fem_ch_3,fem_ch_4,fem_ch_5)
print("t-test p-value:",p)
htestcontrol(p)
#number of children doesnt influence bmi.
# =============================================================================
# RECOMMENDATION
# =============================================================================
# 1.As we can observe the smokers in general have a higher charges so we can create awareness around to stop smoking as it is not at all pocket friendly.
# 2.Women with any number of children have almost same BMI as observed by hypothesis testing, basic awareness around family planning can be provided to keep them from facing financial issues.
# 3.With increasing age the charges too increase, so we can promote a healthy living in the middle ages to avoid these charges in the later stage of life.

# 1.Sigara içenlerin genel olarak daha yüksek ücretleri olduğunu gözlemlediğimiz gibi, sigara hiç cep dostu olmadığı için sigarayı bırakma konusunda çevrede farkındalık yaratabiliriz.
# 2. Herhangi bir sayıda çocuğu olan kadınların, hipotez testi ile gözlemlenen VKİ ile hemen hemen aynı olduğu, aile planlaması konusunda temel farkındalık sağlanarak finansal sorunlarla karşılaşmamaları sağlanabilir.
# 3. Yaşın artmasıyla birlikte ücretler de artar, böylece yaşamın sonraki aşamalarında bu suçlamalardan kaçınmak için orta çağda sağlıklı bir yaşamı teşvik edebiliriz.


















































