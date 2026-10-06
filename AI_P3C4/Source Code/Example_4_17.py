# Example 4.17 CNN Visualize Filters
# Modified from:
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

# ==========================================
# IMPORTS (Dipindahkan ke atas)
# ==========================================
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims

# ==========================================
# INITIALIZATION (Bagian 6 dipindah ke atas agar model dikenali)
# ==========================================
# Load the model
model = VGG19()

# ==========================================
# PART 2: List Semua Layer
# ==========================================
print("--- Layer Summary ---")
n = 0
for layer in model.layers:
    print(n, layer.name)
    n += 1

# ==========================================
# PART 3: List Layer Konvolusi & Weights
# ==========================================
print("\n--- Convolutional Layers Weights ---")
n = 0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)
    n += 1

# ==========================================
# PART 4: Visualisasi Filter (Weights)
# ==========================================
print("\n--- Visualizing Filters ---")
# retrieve weights from the first hidden layer (block1_conv1)
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[2]) # Diperbaiki: index channel pada Keras (H, W, In_C, Out_C)
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 4, 1
plt.figure(figsize=(10,10))
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(s[2]): # Loop berdasarkan jumlah input channel
        ax = plt.subplot(n_filters, s[2], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray') # Menampilkan slice berdasarkan channel
        ix += 1
# show the figure
plt.show()

# ==========================================
# PART 5: Fungsi Plot Feature Maps
# ==========================================
def plot_feature_maps(feature_maps):
    # plot all feature maps
    col = 8
    row = int(feature_maps.shape[3]/col)
    ix = 1
    plt.figure(figsize=(20,20))
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

# ==========================================
# PART 6: Prediksi dan Visualisasi Feature Maps
# ==========================================
print("\n--- Visualizing Feature Maps ---")
# Select the hidden layer to visualize
n = 1
# redefine model to output right after the hidden layer
feature_model = Model(inputs=model.inputs, outputs=model.layers[n].output)
feature_model.summary()

# load the image with the required shape (Pastikan file Elephant.jpg ada di folder yang sama)
img = load_img('Elephant.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)

# get feature map for first hidden layer
feature_maps = feature_model.predict(img)
print("Feature maps shape: ", feature_maps.shape)

# plot all the feature maps
plot_feature_maps(feature_maps)