import numpy as np
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st
from PIL import Image

def load_model():
	return np.load('combined_model_compressed.npz')

def run():

    st.title('First Generation Pokemon Identifier')
    st.subheader('By Braiden Hook')
    st.write('**Find out the identity of your pokemon by uploading their picture!**')

    image_file = st.file_uploader("Upload Pokemon Image", type=["jpg","jpeg"])

    if image_file is not None:
        img = Image.open(image_file)
        st.image(img,width=250)
        st.spinner()
        with st.spinner(text='Please wait as we identify your pokemon  .  .  .'):
        	ans = reverseSearch(image_file)

        print (ans)
        st.write("\nYour Pokemon is most likely a **{}**!".format(ans))
        if ans != "Alolan Sandslash":
            st.write("\nFor more information on your pokemon visit: https://bulbapedia.bulbagarden.net/wiki/{}_(Pokémon)".format(ans))
        else:
            st.write("\nFor more information on your pokemon visit: https://bulbapedia.bulbagarden.net/wiki/Sandslash_(Pokémon)")


def reverseSearch(image_file):

    img_size =224
    model = ResNet50(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3),pooling='max')

    batch_size = 64
    root_dir = 'PokemonData_with_Invert'

    img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    datagen = img_gen.flow_from_directory(root_dir,
                                            target_size=(img_size, img_size),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)

    num_images = len(datagen.filenames)
    num_epochs = int(math.ceil(num_images / batch_size))

    feature_list = load_model()

    print("Num images   = ", len(datagen.classes))
    # print("Shape of feature_list = ", feature_list.shape)

    filenames = [root_dir + '/' + s for s in datagen.filenames]

    neighbors = NearestNeighbors(n_neighbors=5,
                                algorithm='ball_tree',
                                metric='euclidean')
    neighbors.fit(feature_list['arr_0'])


    #bytes_data = image_file.getvalue()
    #img_path = image_file
    input_shape = (img_size, img_size, 3)
    # img = image.load_img(bytes_data, target_size=(input_shape[0], input_shape[1]))
    image = Image.open(image_file)
    img_array = np.array(image)
    img = tf.image.resize(img_array, size=(224,224))
    expanded_img_array = np.expand_dims(img, axis=0)
    copy = np.copy(expanded_img_array)
    preprocessed_img = preprocess_input(copy)


    test_img_features = model.predict(preprocessed_img, batch_size=1)
    _, indices = neighbors.kneighbors(test_img_features)

    def similar_images(indices):
        plt.figure(figsize=(15,10), facecolor='white')
        plotnumber = 1
        results = []   
        for index in indices:
            if plotnumber<=len(indices) :
                # ax = plt.subplot(2,4,plotnumber)
                buffer = os.path.dirname(filenames[index])
                results.append(os.path.basename(buffer))
                # print (filenames[index])
                # plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')            
                plotnumber+=1
        # plt.tight_layout()
        return results

    # print(indices.shape)

    # plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
    # plt.xlabel(img_path.split('.')[0] + '_Original Image',fontsize=20)
    # plt.show()

    results = similar_images(indices[0])
    unique = Counter(results)

    print (results)

    if len(unique) > 1 and unique.most_common(3)[0][1] == unique.most_common(3)[1][1]:
        for name in results:
            if name == unique.most_common(3)[0][0]: return name
            if name == unique.most_common(3)[1][0]: return name

    else: return unique.most_common(3)[0][0]

if __name__=='__main__':
    run()

