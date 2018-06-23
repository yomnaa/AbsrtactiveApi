from flask import Flask, request, jsonify
import os
import sys
import logging
from flask_cors import CORS
from serve import getModelApi  # see part 1.
from Parser import parse
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default
preprocessor_api,model_api = getModelApi()


# logging for heroku
if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)

# API route
@app.route('/api', methods=['POST'])
def api():
    """API function

    All model-specific logic to be defined in the get_model_api()
    function
    """
    summaries=[]
    articles=[]
    input_data = request.json
    i=0
    for url in input_data:
        app.logger.info("api_input: " + str(url))
        p=preprocessor_api(parse(url))
        # open('art'+str(i),'w').write(p)
        articles.append(p)
        app.logger.info("article text: " + articles[-1])
        i+=1

    summaries= model_api(articles)
    
    app.logger.info("summariescount: " + str(len(summaries))+" "+str(len(articles)))

    response = jsonify(summaries)
    return response

# default route
@app.route('/')
def index():
    return "Index API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
