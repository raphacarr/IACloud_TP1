from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Web App TD IA Cloud M2 Carrilho !"

@app.route('/formulaire')
def pageHtml():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    
