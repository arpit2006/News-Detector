import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

#Load Dataset
fake = pd.read_csv("D:/Data Science/ML Model/New Detector/fake.csv")
true = pd.read_csv("D:/Data Science/ML Model/New Detector/true.csv")

#Give Labels
fake["labels"] = 0
true["labels"] = 1

data = pd.concat([fake,true])
data = data.dropna()
# print(data)

#Shuffle data
data = data.sample(frac=1,random_state=42)

# print(data.head())
# print(data.tail())
# print(data.info())

null_val = data.isnull().sum()
# print(null_val)

#Initialize variables
# Text Cleaning
data["title"] = data["title"].str.lower()
data["text"] = data["text"].str.lower()

data["title"] = data["title"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
data["text"] = data["text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

# Initialize variables
x = data["title"] + " " + data["text"]
y = data["labels"]

#train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

#Convert text to model
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.75,
    min_df=2,
    ngram_range=(1,2)
)

x_train_prepared = vectorizer.fit_transform(x_train)
x_test_prepared = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_prepared,y_train)
log_predictions = model.predict(x_test_prepared)
print("Accuracy is: ",accuracy_score(y_test,log_predictions) * 100)

model1 = PassiveAggressiveClassifier(max_iter=1000)
model1.fit(x_train_prepared,y_train)
passive_predictions = model1.predict(x_test_prepared)
print("Accuracy is: ",accuracy_score(y_test,passive_predictions) * 100)


news = ["The United States President met with NATO leaders in Brussels to discuss regional security and economic cooperation."]
new_vec = vectorizer.transform(news)
predictions = model1.predict(new_vec)
if(predictions[0] == 0):
    print("False")
else:
    print("True")

news = [
    "The Indian Space Research Organisation (ISRO) successfully launched a new weather monitoring satellite from Sriharikota on Friday."
    ]
new_vec = vectorizer.transform(news)
predictions = model.predict(new_vec)
if(predictions[0] == 0):
    print("False")
else:
    print("True")





