# This file together with the chatbothtml.html file
# Enables a button on the website which will call the
# Chatbot.predict() function and run test
from flask import Flask, request, render_template
import chatbot
app = Flask(__name__)


@app.route('/ChatBot_home/')
def index():
    return render_template("chatbothtml.html")


@app.route('/ChatBot_home/test')
def test():
    chatbot.predict()
    return '<h3>Hey the ChatBot program is running now.</h3>'

if __name__ == '__main__':
    app.run(debug=True)