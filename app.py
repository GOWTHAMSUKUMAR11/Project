from flask import Flask,render_template,request,url_for
from imgEncryptionandDecryption import imageprocess

app = Flask(__name__)

image_names = ["1_permuted_image.png", "2_col_permuted_image.png", "3_dct_image.png", "4_final_encrypted_img.png", "5_decrypted_image_from_ps.png", "6_idct_image.png", "7_reverse_col_permuted_image.png", "8_final_decrypted_image.png"]


@app.route('/')
def index():
    return render_template("index.html")
@app.route('/',methods = ['POST'])
def imageencryption():
    if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                image.save('static/uploads/' + "bwimages.jpg")
    imageprocess.imageEncryptAndDecrypt()
    return render_template('slideshow.html', image_names=image_names)

if __name__ == "__main__":
    app.run(debug=True)