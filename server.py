# This is a simple API created to run ChatBot.py
# By return a notification message and execute the script
import chatbot
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class ExecuteChatBot(Resource):
    def get(self):
        chatbot.predict()
        return {'message': "The ChatBot is Running!"}

api.add_resource(ExecuteChatBot, '/ChatBot/Test')

if __name__ == "__main__":
    app.run(port=5002)