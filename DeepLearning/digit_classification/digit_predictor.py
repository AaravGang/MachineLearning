import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2

# dataset of handwritten digits
# comes in as ((train_x,train_y),(test_x,test_y))
mnist = keras.datasets.mnist.load_data()
train_data, test_data = mnist
print(train_data)
train_images, train_labels = train_data
test_images, test_labels = test_data

# image constants
IMAGE_DIM = 28
IMAGE_SHAPE = (IMAGE_DIM, IMAGE_DIM)
TOTAL_PIXELS = IMAGE_DIM ** 2
TOTAL_CLASSES = 10


def view_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(labels[i])
    plt.show()


# view_images(train_images,train_labels)

# each image is 28 x 28, 2d array
# each element in the 28 x 28 is a gray scale value, from 0-255
# scale this down to a range of 0-1
train_images = train_images / 255
test_images = test_images / 255

# view_images(train_images,train_labels)


"""
So how does this Neural Network work?
we have an input layer, some hidden layers and an output layer
input > HL1 > HL2 > ... > HLn > output

Each layer may have a variable number of nodes
In each hidden layer, the following happens in each node:

# feed-forward algorithm
takes in an input array > dot products them with the weights > adds the biases > applies an activation function > sums the new formed array and returns this value

In the output layer, however the exact same thing happens, other than the applying of activation function

to tweak the weights and the biases, we back-propagate

One cycle of this process is one epoch

"""


def create_model():
    # set up the layers for the model
    # sequential does one step after the another, output of one layer will be input to another
    model = keras.Sequential(
        [
            # the input layer - it shall be a flattened array
            # tf.keras.layers.Flatten transforms a 2d array to a 1d array
            keras.layers.Flatten(input_shape=IMAGE_SHAPE),
            # hidden layer 1 - Dense -> all nodes are connected to all other nodes
            # number of nodes, activation function
            keras.layers.Dense(512, activation="relu"),
            # the output layer
            # 10 nodes, each node contains a score for each of the available classes
            # i.e, [0.05,0.94,0.0001,0.0004... ] - kinda thing
            keras.layers.Dense(TOTAL_CLASSES),
        ]
    )
    
    
    
    # compile the model
    
    # Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
    # Optimizer —This is how the model is updated based on the stokes_data it sees and its loss function.
    # Metrics —Used to monitor the training and testing steps. accuracy - the fraction of the images that are correctly classified.
    
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    return model


# train the model
def fit(model, train_features, train_labels, epochs=10, batch_size=100, verbose=2, save=False):
    # each epoch is a cycle of feed-forward and back-propagation
    # batch size is, what fraction of the total stokes_data do we want to train on at one time
    # this helpful, cuz we don't have to load the entire stokes_data in ram at once
    # size of each batch will be len(training_data/batch_size)
    print("Training...")
    model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    if save:
        model.save("digit_classifier")


# calculate the accuracy of the model
def score(model, test_features, test_labels, verbose=2):
    print("Testing...")
    test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=verbose)
    print("Test accuracy:", test_acc, ", Test loss: ", test_loss)
    return test_acc, test_loss


# predict given an array of test_features, [test_features_1,test_features_2 ... ]
def predict(model, test_features):
    # make predictions
    # this model will output the probabilities for the given input to be classified into a group
    prediction_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = prediction_model.predict(np.expand_dims(test_features[0], 0))
    return tf.argmax(predictions[0]), predictions


# load the model from a saved version, or create one
def load_model(saved=True, features=None, labels=None):
    if saved:
        return keras.models.load_model("digit_classification/digit_classifier")
    
    model = create_model()
    fit(model, features, labels, save=True)
    return model


# pygame stuff from here on out

class Colors:
    WHITE = (255, 255, 255)
    GREY = (153, 153, 53)
    BLACK = (0, 0, 0)
    YELLOW = (250,250,25)
    CYAN = (25,250,250)
    PURPLE = (180,40,90)


class Spot:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.color = Colors.BLACK
        self.width = width
    
    def get_pos(self):
        return self.row, self.col
    
    def reset(self):
        self.color = Colors.BLACK
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
        pygame.draw.rect(win, Colors.PURPLE, (self.x, self.y, self.width, self.width), 1)


# init a grid
def make_grid(rows, window_width):
    grid = []
    gap = window_width / rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap)
            grid[i].append(spot)
    
    return grid


# resize the image
def get_resized(pixels, dim):
    resized_image = cv2.resize(pixels, (dim, dim))
    return resized_image


# view an image along with the grid
def view_image(win, grid, image):
    win.fill(0)
    
    for r, row in enumerate(grid):
        for c, spot in enumerate(row):
            spot.color = image[r][c]
            spot.draw(win)
    
    pygame.display.update()


# get the pixels of the screen
def get_pixels(win, n_rows, n_cols):
    pixels = []
    for y in range(n_rows):
        pixels.append([])
        for x in range(n_cols):
            pixels[y].append(win.get_at((x, y)))
    
    return np.array(pixels)


# function to crop the image to a tight fit
def crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(gray)  # Find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    return rect


def predict_drawing(win, screen_dim, center_img_dim, n_rows, model):
    pixels = get_pixels(win, screen_dim, screen_dim)  # get all the pixels on the screen
    pixels = crop(pixels)  # crop it to fit exactly
    center_image = get_resized(pixels, center_img_dim)  # get a resized 20x20 image
    
    # create a 28x28 image which has that 20x20 image in the center
    image = []
    for i in range(n_rows):
        image.append([])
        for j in range(n_rows):
            if i < (n_rows - center_img_dim) / 2 or i >= center_img_dim + (n_rows - center_img_dim) / 2 or j < (
                    n_rows - center_img_dim) / 2 or j >= center_img_dim + (n_rows - center_img_dim) / 2:
                image[i].append([0, 0, 0, 0])  # [r g b a]
            else:
                image[i].append(center_image[i - 4][j - 4])
    
    # view this image
    grid = make_grid(n_rows, screen_dim)
    view_image(win, grid, image)
    
    # each pixel is a an array of [r g b a], make it gray scale scalar value
    image = np.average(image, axis=2)
    
    # scale this down to a range of 0.0 and 1.0 - cuz this is what we did for the training set
    image = np.array(image) / 255
    
    prediction = predict(model, [image])
    return prediction


def main():
    model = load_model(saved=True, features=train_images, labels=train_labels)
    # score(model, test_images, test_labels)
    
    # screen constants
    n_rows = IMAGE_DIM
    screen_dim = TOTAL_PIXELS
    center_img_dim = 20
    prev_mouse = (0, 0)
    
  
    win = pygame.display.set_mode((screen_dim, screen_dim))
    win.fill(0)
    pygame.display.update()
    pygame.display.set_caption("Digit Classifier")

    pygame.font.init()
    font = pygame.font.Font('freesansbold.ttf', 32)
    
    run = True
    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                break
            
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    prediction = predict_drawing(win, screen_dim, center_img_dim, n_rows, model)
                    print("predicted: ", prediction[0].numpy())

                    text = font.render(f'Predicted: {prediction[0].numpy()}', True, Colors.BLACK,Colors.CYAN)
                    textRect = text.get_rect()
                    textRect.center = (screen_dim // 2, 100)
                    win.blit(text,textRect)
                    pygame.display.update()
                
                if e.key == pygame.K_c:
                    win.fill(0)
                    pygame.display.update()
        
        # draw
        if pygame.mouse.get_pressed()[0]:  # left mouse
            pos = pygame.mouse.get_pos()
            pygame.draw.line(win, (255, 255, 255), prev_mouse, pos, 30)
            pygame.display.flip()
        
        # erase
        if pygame.mouse.get_pressed()[2]:  # right mouse
            pos = pygame.mouse.get_pos()
            pygame.draw.line(win, (0, 0, 0), prev_mouse, pos, 20)
            pygame.display.flip()
        
        prev_mouse = pygame.mouse.get_pos()  # set the previous mouse location


main()




