# Scene_recogniton_Using_CNN
Scene recognition involves detection of scene in an image. This topic has received great deal of attention in computer vision because of its wide applications such as it’s an important feature for driver-less car. To detect the scene accurately, we are going to use Convolutional neural network (CNN). Neural network is a computational model that works in a similar way to the neurons in the brain. Each neuron takes an input, performs some operations then passes the output to the following neuron. Convolutional neural network is a type of artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex. The visual cortex has small regions of cells that are sensitive to specific regions of visual field. Some individual neuronal cells in brain responds only in presence of edges of a certain orient action. For example, neurons fires when exposed to vertical edges and some when shown horizontal or diagonal edges. We are going to teach the computer to recognize the scene in image and classify them into one of the 6 categories such as auditorium, farms, labs etc. For example, we need to recognize the cat in image. To do so, we first need to teach the computer how a cat looks like before it being able to recognize a new object. The more cats the computer sees, the better it gets in recognizing cats. With the help of CNN, the computer will start recognizing the patterns present in cat pictures that are absent from other ones and will start building its own cognition. Convolutional neural network (CNN) is one of the most popular techniques used in improving the accuracy of image classification. CNN has convolution layer at the beginning which breaks the image into number of tiles, the machine then tries to predict what each tile is. Finally, the computer tries to predict what’s in the picture based on the prediction of all the tiles. Computer extract features from each tile that are called intra scale features. To detect the scene, we need to combine all the intra scale features to form the multi scale feature.
The project takes as captured image as input and predicted the accurate scene using Convolutional neural network.
Technology: Python, Tesnsorflow 1.8, Keras Api, Django framework and other deep learning libraries.


Scene Categeries:
auditorium
bathroom
bedroom
farm
forest
Swimming pool
