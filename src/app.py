from flask import Flask, render_template, send_from_directory
import os



app = Flask(__name__)


# app.config['UPLOAD_FOLDER'] = 'C:\\Users\\MJR0X3R\\PycharmProjects\\flaskapp\\uploads' 


@app.route("/")  # the main route
def home():
    return render_template("index.html")


@app.route('/figures/<path:filename>')
def figures(filename):
    figures_dir = os.path.join('C:\\','Users','MJR0X3R','DataScienceProjects','imbalanced_project','reports','figures','testing')
    return send_from_directory(figures_dir, filename)



if __name__ == "__main__":
    app.run(host='10.252.98.30')
    # app.run(debug=True)

