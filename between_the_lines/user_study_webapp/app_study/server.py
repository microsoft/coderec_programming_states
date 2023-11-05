from bottle import route, run, static_file
import bottle
import threading
import json
import pickle
import  argparse
import re
from time import sleep
import os
import random
parser = argparse.ArgumentParser()

app = bottle.Bottle()
index = -1
session_id = 0
logs = ""
my_module = os.path.abspath(__file__)
parent_dir = os.path.dirname(my_module)
static_dir = os.path.join(parent_dir, 'static')
json_path = ""
video_path = ""
@app.get("/")
def home():
    with open('index.html', encoding='utf-8') as fl:
        html = fl.read()
        return html

@app.get('/static/<filename>')
def server_static(filename):
    return static_file(filename, root=static_dir)
@app.post('/update_json')
def update_json():
    # receive json from request
    data = bottle.request.json
    # save data to json file
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)
    return

@app.post('/initialize_html')
def initialize_html():
    # receive json from request
    # load json file
    with open(json_path, 'r') as fl:
        data = json.load(fl)
    logs = data['logs']
    labeled_so_far = 0
    for i in range(len(logs)):
        if logs[i]['label'] != 'not_labeled':
            labeled_so_far += 1
    data_json = {'json_path': json_path, 'video_path': video_path, 'labeled_so_far': labeled_so_far}
    return  data_json

class Demo(object):
    def __init__(self):
        run_event = threading.Event()
        run_event.set()
        self.close_thread = True
        threading.Thread(target=self.demo_backend).start()
        app.run(host='localhost', port=8080)
        try:
            while 1:
                pass
        except KeyboardInterrupt:
            print("Closing server...")
            self.close_thread = False

    def demo_backend(self):
        global query, response
        while self.close_thread:
            sleep(0.01)
            pass
            #if query:
            #    response = self.model.ask(query)
            #    query = []

parser.add_argument('-p', '--path', help='Path to logs json', required=False) # change to True
parser.add_argument('-v', '--video', help='Path to video', required=False) # change to True

def main():
    args = parser.parse_args()
    global session_id, json_path, video_path
    json_path = args.path
    video_path = args.video
    #session_id = int(args.session)
    demo = Demo()

if __name__ == "__main__":
    main()
# python server.py -p static/test.json -v static/july3-4pm-linearreg_accepts_trim.mp4