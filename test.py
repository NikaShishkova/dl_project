from model import *
from data import *
from PIL import Image
from imagehash import dhash

testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
new_model1 = load_model('new_model.hdf5')
results = new_model1.predict_generator(testGene, 5, verbose=1)
saveResult("data/membrane/test", results)
flag = True
for i in range(5):
    image1 = dhash(Image.open('data/membrane/test/' + str(i) + '_predict.png'))
    image2 = dhash(Image.open('test_png/' + str(i) + '_test.png'))
    if image1 != image2:
        flag = False
        break
if flag:
    print('Tests passed :)')
else:
    print('Something went wrong :(')