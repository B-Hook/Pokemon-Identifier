# Pokemon-Identifier
## By Braiden Hook

### Goal

To create a publicly available machine learning resource with the ability to read in an image of a first-generation Pokémon and return its identity (name).


### App Link

For Streamlit Cloud (Recommended): https://share.streamlit.io/b-hook/pokemon-identifier/main/app.py

For AWS (has memory issues so it is slower): http://35.88.142.36:8501/

### Model

To accomplish this goal, I implemented a TensorFlow (keras) model, ResNet50, which is a convolutional neural network. I used this model mainly to classify the images in the dataset that I loaded in. Following that I used a sklearn model, NearestNeighbors (KNN), which combined with the ResNet50 model, allowed me to locate the five images that were most similar to the inputted image. Through that, I simply selected the Pokémon that appeared most out of those images and declared that as its identity.

Original notebook: https://www.kaggle.com/code/ajax0564/reverse-image-search-with-tensorflow-2-2/notebook

ResNet50: https://viso.ai/deep-learning/resnet-residual-neural-network/

KNN: https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

### Dataset 

The model was trained on a dataset with 6837 images of first-generation Pokémon. Below is a link to where I found the dataset.

Original dataset: https://www.kaggle.com/datasets/lantian773030/pokemonclassification

### Extension

To improve the quality of this model, I decided to expand on the dataset I loaded in by inverting all the images and training the model off a combination of them both. To succeed in doing this I used the PIL (pillow) library to invert all the images as I did a transversal over all the files. Unfortunately, it was not possible to invert all the images due to compatibility errors. Out of the 6837 images, 127 could not be inverted, I suppose it could have been worse.

I did tests on just using the invert images to train the model as well as a combination of the non-inverted and inverted. The inverted was less accurate than the original by themselves, but it still returned similar results nonetheless. 

After combining the two datasets to create a dataset of 13531 images (PokemonData_with_Invert), I used it to train the model where it returned a mixed of non-invert and invert (obviously). However, I was disappointed in that it was hard to see a huge difference between the original and combined datasets. This is mainly because there was still the same amount of specific Pokémon. Nonetheless, I was still satisfied with the performance and decided to use it on my final app.

### Issues That Arose

#### Memory

Following the testing stage, I was confident in where things were heading. However, I regretted my thought process after navigating through this stage. The first problem that occurred was the models being too large. To decrease the time I saved a numpy array that I planned to load into the program. However, this file was too large and required git lfs. Little did I know, Heroku, the web service I planned to use to launch my app, did not support it. After realizing that problem I tried using AWS, but whenever I tried to load in my array it would kill the process as it ran out of memory. In the end, I decided to compress the numpy array to save as much space as possible and avoid using git lfs. Unfortunately, this did not provide a fix to my problem. Luckily, streamlit cloud was able to host my app and it ran the way it was supposed to.

I was also able to find a fix for AWS, but at the cost of performance. It is due to this that I recommend using the app on streamlit cloud.

Freeing Extra Memory in AWS: https://stackoverflow.com/questions/17173972/how-do-you-add-swap-to-an-ec2-instance

#### Image Types

The other problem that I encountered was utilizing streamlit’s uploader (for images). Due to incompatibility, I was unable to load in the image through TensorFlow (keras) and instead had to utilize pillow. As consequence for this I became unable to use many of the image data types that I was able to use while testing. The reasons for this were due to the lack of compatibility between ResNet50 and pillow. I did not have to worry about this during testing as both ResNet50 and the image loading process were part of the same library.

This is the reason why only jpg/jpeg is the only input file allowed to be uploaded on the app.

### Conclusion

In conclusion, I was able to learn a lot from this project. I learned how to process and classify images. As well as reverse image search them and return five similar images. I also was able to get more familiar with memory management. As well as come up with alternatives to unexpected problems. It was exciting to be able to work on something that I held an interest in. It has also encouraged me to do something similar once again where I can focused on more interests of mine.




### Resources Used
Pokemon Dataset: https://www.kaggle.com/datasets/lantian773030/pokemonclassification

Reverse Image Search Code: https://www.kaggle.com/code/ajax0564/reverse-image-search-with-tensorflow-2-2/notebook

### Additional Readings

ResNet50: https://viso.ai/deep-learning/resnet-residual-neural-network/

KNN: https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75


