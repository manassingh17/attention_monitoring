import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X, y = joblib.load("embeddings.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["attentive", "inattentive"]))

joblib.dump(clf, "attention_classifier.pkl")
print("Model saved to attention_classifier.pkl")
