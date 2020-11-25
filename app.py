from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from models import base64str_to_PILImage , PILImage_to_base64str,resize_image,preprocess_image,prediction


path="path to saved model"

app = Flask(__name__)
api = Api(app)


parser = reqparse.RequestParser()
parser.add_argument('base64',help='send in the base 64 code of the image here')



class Predict(Resource):
    def get(self):

        data = parser.parse_args()
        input_base64=data["base64"]

        image=base64str_to_PILImage(input_base64)
        
        model=self.import_model()

        mask=prediction(image,model)
        class_list = ["Buildings", "Misc.Manmade structures" ,"Road",\
                  "Track","Trees","Crops","Waterway","Standing water",\
                  "Vehicle Large","Vehicle Small"]
    
        img = np.zeros((image.shape[0],image.shape[1],3))
        img[:,:,0] = image[:,:,4] #red
        img[:,:,1] = image[:,:,2] #green
        img[:,:,2] = image[:,:,1] #blue

        for i in range(num_cls):
            plt.figure(figsize=(25,25))
            ax1 = plt.subplot(131)
            ax1.set_title('image ID:6120_2_0')
            ax1.imshow(adjust_contrast(img))
            ax2 = plt.subplot(132)
            ax2.set_title("predict "+ class_list[i] +" pixels")
            ax2.imshow(mask[i], cmap=plt.get_cmap('gray'))
            plt.show()




        return {'hello': 'lokesh'}

    def import_model():
        '''
        importing the model here
        '''
        model = load_model('unet.h5', custom_objects={"jaccard_coef": jaccard_coef})

        return model



api.add_resource(Predict, '/Predict')

if __name__ == '__main__':
    app.run(debug=True)