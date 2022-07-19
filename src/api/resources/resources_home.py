
from flask import render_template, make_response
from flask_restful import Resource


class HomeResource(Resource):
    def __init__(self):
        pass

    def get(self):
        headers = {'Access-Control-Allow-Origin':'*'}
        template = render_template('plugin.html')
        return make_response(template,200,headers) 