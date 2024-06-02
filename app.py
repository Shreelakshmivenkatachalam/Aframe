from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        data = pd.read_csv(file)
        data = data.dropna()

        X = data.drop('Sales', axis=1)
        y = data['Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=200, random_state=62)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head().to_html()

        importances = model.feature_importances_
        feature_names = X.columns
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(
            by='Importance', ascending=False).to_html()

        return render_template('results.html', mse=mse, r2=r2, comparison=comparison,
                               feature_importances=feature_importances)


if __name__ == '__main__':
    # Debug statements to print the current working directory and check template existence
    print("Current Working Directory:", os.getcwd())
    print("Index template exists:", os.path.exists('templates/index.html'))
    print("Results template exists:", os.path.exists('templates/results.html'))

    app.run(debug=True)
