import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pygame, cv2, copy, pickle
import pandas as pd

NUM_CLASSES = 10 # 0-9
IMAGE_SHAPE = (28, 28, 1) # [[[],[],[]...x28],[],...x28]
IMAGE_SHAPE_2D = (28, 28)
DIM = 28


with open("data/1st_three", "rb") as f:
    dummy_set = pickle.load(f)


def load_data():
    train_dataset = pd.read_csv("data/emnist-digits-train.csv", )
    train_labels = train_dataset.values[:, 0]  # the first column is the labels

    # reshape into format that the cnn wants. Each image comes in as a linear array of 784 gray values
    # transpose, cuz x and y of this thing is messed up - or mine maybe...
    train_set = train_dataset.values[:, 1:].reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)

    test_dataset = pd.read_csv("data/emnist-digits-test.csv")
    test_labels = test_dataset.values[:, 0]
    test_set = test_dataset.values[:, 1:].reshape(-1, 28, 28, 1).transpose(0, 2, 1, 3)

    print("Loaded datasets")
    return train_set, train_labels, test_set, test_labels


# train_set,train_labels,test_set,test_labels = load_data()


def view_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(min(len(images), 25)):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(labels[i])
    plt.show()


# view_images(test_set,test_labels)

# each image is now a 28 x 28, 2d array
# each element in the 28 x 28 is a gray scale value, from 0-255
# scale this down to a range of 0-1

# train_set = train_set / 255
# test_set = test_set / 255

# HOW DOES A CONVOLUTION NEURAL NETWORK WORK????????
# https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
"""
A CNN has 3 main layers:
Convolution layer, Pooling layer, and a fully connected layer.

CONV LAYER
A kernel is just a matrix of learn-able parameters. The kernel slides over
the image, creating an activation map. Usually the values in activation map,
are dot products of the correspondig region of the image and the kernel itself.

In a normal neural network, every output unit interacts with every input unit.
In a ConvNet, there is only sparse interaction. The size of the image maybe 1000x1000. But the
kernel might be just 9x9, which allows it to extract features in a specific region.

To construct one activation map, neurons are restricted to use only one set of weights.
This is not the case in dense networks.

POOLING LAYER
Pooling layers are used to reduce the spatial size.
In max pooling, the maximum value is chose from a given region.
Pooling is done on the activation map created by the conv layer.

FULLY CONNECTED LAYER
A typical dense network, with weights and biases. Helps to relate b/w input and output.
Activation ReLU - Rectified Linear Unit = max(0,k), for any value k.
More reliable in CNNs than sigmoid, which squashes the value b/w 0 and 1,
and tanh, that squashes the value b/w -1 and 1.
"""


def create_model():
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=32,  # number of filters
                kernel_size=(3, 3),  # kernel dimensions
                # pad the matrix with zeros, to get a uniform one.
                # "same" will keep the output volume same as input volume.
                # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
                padding="same",
                input_shape=IMAGE_SHAPE,
                activation="relu",  # gets rid of all negative values, makes image sharper
            ),
            keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            ),  # reduce image dimensions... iggg
            keras.layers.Flatten(),  # flatten for dense to work on it
            # 128 is number of nodes,
            # relu gets rid of all -ve numbers - thereby getting rid of all unwanted details
            keras.layers.Dense(128, activation="relu"),
            # output layer - softmax for probabilities
            keras.layers.Dense(
                NUM_CLASSES, activation="softmax"
            ),  # output probabilities
        ]
    )

    # compile model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # research!
        # optimizer used to minimize loss - just a math function!
        optimizer=keras.optimizers.SGD(
            learning_rate=0.01
        ),  # gradient descent - research!
        metrics=["accuracy"],  # what do we care about? -iggg
    )

    return model


def train(model, images, labels, batch_size=100, epochs=3, verbose=1, save=True):
    print("Training...")

    # images.shape = (n,28,28,1)
    model.fit(
        images, labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
    )

    if save:
        model.save("digit_classifier_cnn")


def evaluate(model, images, labels, verbose=1):
    print("Testing...")

    # images.shape = (n,28,28,1)
    test_loss, test_acc = model.evaluate(images, labels, verbose=verbose)
    print("Test accuracy:", test_acc, ", Test loss: ", test_loss)
    return test_acc, test_loss


def load_model(saved=True):
    if saved:
        return keras.models.load_model("digit_classifier_cnn")

    return create_model()


# cnn_classifier = load_model()
# train(cnn_classifier,train_set,train_labels)
# evaluate(cnn_classifier,test_set,test_labels)


# pygame stuff from here
win = pygame.display.set_mode((DIM ** 2, DIM ** 2))
win.fill(0)
pygame.display.update()
pygame.display.set_caption("Convolution Digit Classifier")
pygame.font.init()
font = pygame.font.SysFont("Arial", 32,)


def draw_image(win, image):
    win.fill(0)
    image = (
        image.reshape(IMAGE_SHAPE) * 255
    )  # image is gray scale image {0,1}, make it {0,255}
    rect_dim = win.get_width() / DIM
    for i in range(DIM):
        for j in range(DIM):
            pygame.draw.rect(
                win,
                (image[j][i],) * 3,
                (i * rect_dim, j * rect_dim, i * (rect_dim + 1), j * (rect_dim + 1)),
            )

    pygame.display.update()


def write(win, text):
    text_render = font.render(text, False, (20, 250, 250))
    text_rect = text_render.get_rect(center=(win.get_width() / 2, 50))
    win.blit(text_render, text_rect)
    pygame.display.update()


# resize the image
def get_resized(image, shape):
    resized_image = cv2.resize(image, shape)
    return resized_image


# get the pixels of the screen
def get_pixels(win):
    pixels = copy.deepcopy(pygame.surfarray.pixels_blue(win)).T
    return pixels


# function to crop the image to a tight fit
def crop(img):
    # img will be a "grayscale" image
    coords = cv2.findNonZero(img)  # Find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[
        y : y + h, x : x + w
    ]  # Crop the image - note we do this on the original image
    return rect


def blur(img, t="filter"):
    if t == "filter":
        # Apply blurring kernel
        kernel = np.ones((3, 3), np.float32) / 9
        return cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    elif t == "gaussian":
        # apply gaussian blur on src image
        dst = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
        return dst


def apply_filters(image):
    cropped = crop(image)  # crop the image and get only the drawing
    resized_image = get_resized(cropped, (DIM - 4, DIM - 4))  # resize into 20x20
    final_resized = np.zeros(IMAGE_SHAPE_2D, dtype=np.float32)
    final_resized[2:-2, 2:-2] = resized_image  # place inside 28x28

    blurred = blur(
        final_resized, "gaussian"
    )  # blur the image, to make it like the ones which are in training set

    final_images = blurred.reshape(
        1, DIM, DIM, 1
    )  # reshape it to be inside an outer array, as the cnn wants it to be that way
    final_images = final_images / 255  # normalise the values b/w 0 and 1

    return final_images


def predict_drawing(model, images):
    probabilities = model.predict(
        images
    ).flatten()  # probabilities will be [[0.1,0.85,0.03,....]]
    predicted = np.argmax(probabilities)

    return predicted, probabilities[predicted], probabilities


def main(model):

    run = True
    prev_mouse = (0, 0)
    draw_image(win, dummy_set[0].reshape(28, 28) / 255)
    
    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                break

            if e.type == pygame.KEYDOWN:
                # clear
                if e.key == pygame.K_c:
                    win.fill(0)
                    pygame.display.update()

                # predict
                if e.key == pygame.K_RETURN:
                    pixels = get_pixels(win)
                    new_image = apply_filters(pixels)

                    draw_image(win, new_image)

                    prediction, confidence, probabilities = predict_drawing(
                        model, new_image
                    )
                    print(
                        f"Predicted: {prediction}| probability: {confidence}   {probabilities}"
                    )
                    write(win, f"( {prediction}, {confidence} )")

        # draw
        if pygame.mouse.get_pressed()[0]:  # left mouse
            pos = pygame.mouse.get_pos()
            pygame.draw.line(win, (255, 255, 255), prev_mouse, pos, 50)
            pygame.display.flip()

        # erase
        if pygame.mouse.get_pressed()[2]:  # right mouse
            pos = pygame.mouse.get_pos()
            pygame.draw.line(win, (0, 0, 0), prev_mouse, pos, 50)
            pygame.display.flip()

        prev_mouse = pygame.mouse.get_pos()  # set the previous mouse location


if __name__ == "__main__":
    classifier = load_model(saved=True)
    main(classifier)
