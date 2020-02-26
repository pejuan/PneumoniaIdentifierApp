import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, random, string
import numpy as np
import io
import os

from PIL import Image
from base64 import b64encode
from tornado.options import define, options
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input

define("port", default=8888, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/images/(.*)",tornado.web.StaticFileHandler,{'path':'./images'}),
            (r'/(favicon.ico)', tornado.web.StaticFileHandler, {"path": ""}),
            (r"/upload", UploadHandler)
        ]
        tornado.web.Application.__init__(self, handlers)
        
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("upload.html")

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        
        model = load_model('models/ds-32-20-100sz.h5',compile=False)
        file1 = self.request.files['xray'][0]
        original_fname = file1['filename']
        extension = os.path.splitext(original_fname)[1]
        fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(12))
        final_filename= fname+extension
        output_file = open("images/" + final_filename, 'wb')
        output_file.write(file1['body'])
        output_file.close()

        path = 'images/'+final_filename
        img = load_img(path,target_size=(100,100))
        im = img_to_array(img)
        x = preprocess_input(np.expand_dims(im.copy(), axis=0))
        pred_class = model.predict_classes(x)
        pred = model.predict(x)

        result = ""
        likelihood = 0
        alert = ""
        if pred_class[0] == 0:
            result = "No Pneumonia"
            likelihood = round(pred[0][0]*100,2)
            alert = "alert-success"
        else:
            result = "Pneumonia"
            likelihood = round(pred[0][1]*100,2)
            alert = "alert-danger"
        print("Result: ", result)
        print("Likelihood: ", likelihood)
        
        message = result+" was found with a "+str(likelihood)+"% "+" confidence."

        img_file = open(path, "rb")
        strForEncode = b64encode(img_file.read())
        img_file.close()
        
        encodedIm = strForEncode.decode('utf-8')
        mime = "image/"+extension[1:]+";"
        displayIm = "data:%sbase64,%s" % (mime,encodedIm)
        
        os.remove(path)

        self.render('result.html', msg=message, alert=alert, pic=displayIm)
        
def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
    
if __name__ == "__main__":
    main()