from flask import Flask
from flask import request

app = Flask(__name__)
@app.route('/')
def index(name='Treehouse'):
  name=request.args.get('name',name)
  #return "Hello from {}".format(name)
  return 'Hello '+name
app.run(debug=True, port=888, host='0.0.0.0')