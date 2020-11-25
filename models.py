import base64
import io
from PIL import Image

def base64str_to_PILImage(base64str):
   base64_img_bytes = base64str.encode('utf-8')
   base64bytes = base64.b64decode(base64_img_bytes)
   bytesObj = io.BytesIO(base64bytes)
   img = Image.open(bytesObj)
   return img

def PILImage_to_base64str(full_img_path):
    with open(full_img_path, "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")
        return base64str


def resize_image(image, model):
      """
  to resize the image
  """
  if image.shape == (837,837,8):
    return image

  else:
    resized_data = resize(image, (837,837,8))
    imwrite('resized.tif', resized_data, planarconfig='CONTIG')
    return tiff.imread("resized.tif")


#https://www.kaggle.com/aamaia/rgb-using-m-bands-example

def adjust_contrast(bands, lower_percent=2, higher_percent=98):
    """
    to adjust the contrast of the image 
    bands is the image 
    """
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)


def prediction(image, model):

    """
    take inputs as image id and trained model 
    and returns predicted mask 
    """
  
    #Read 16-band image
    # rgb_img = os.path.join( 'sixteen_band', '{}_M.tif'.format(id))      
    # rgb_img = tiff.imread(rgb_img)
    rgb_img=image
    img = np.rollaxis(rgb_img, 0, 3)


    #resize the image according to model architecture
    img = resize_image(img, model)

    #adjust the contrast of the image
    x = adjust_contrast(img)
    
    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((num_cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x
     
    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * size:(i + 1) * size, j * size:(j + 1) * size])
            
  
        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
       # x = 2 * np.transpose(line, (0, 1, 2, 3)) - 1

        tmp = model.predict(x, batch_size=4)
       # tmp = np.transpose(tmp,(0,3,1,2))

        for j in range(tmp.shape[0]):
            prd[:, i * size:(i + 1) * size, j * size:(j + 1) * size] = tmp[j]
     
    # thresholds for each class 
    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(num_cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]


def test_lokesh():
    print("testing imoport !!!!!")