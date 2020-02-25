import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
import numpy as np
import io

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

        path = 'images/'+final_filename
        img = load_img(path,target_size=(100,100))
        im = img_to_array(img)
        x = preprocess_input(np.expand_dims(im.copy(), axis=0))
        pred_class = model.predict_classes(x)
        pred = model.predict(x)

        result = ""
        likelihood = 0
        if pred_class[0] == 0:
            result = "Normal"
            likelihood = pred[0][0].round(4)*100
        else:
            result = "Pneumonia"
            likelihood = pred[0][1].round(4)*100
        print("Result: ", result)
        print("Likelihood: ", likelihood)

        self.render('result.html',imurl=path, pred = result, score = likelihood)
        
def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
    
if __name__ == "__main__":
    main()