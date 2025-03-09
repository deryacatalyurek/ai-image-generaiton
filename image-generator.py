import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
import os

# Suppress TensorFlow deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


class ImageGenerator:
    def __init__(self):
        print("Importing 2 image paths... \n")

        self.content_img_path = r"C:\Users\derya\Desktop\ai-image-generaiton\img.jpeg"
        self.style_img_path = r"C:\Users\derya\Desktop\ai-img-gen\style.jpg"

        print(f"Content image - to be converted into a painting: {self.content_img_path} \n")
        print(f"Style image:  {self.style_img_path}\n")

        
        print("Generating your image...")
        self.content_layer = "block5_conv2"
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

        #Build and store the VGG19 model
        self.model = self.build_vgg_model()

        print('Optimizing!\n')
        self.optimization_loop()

    def load_and_process_image(self, path):
        """
        Load an image, resize it, convert to an array, and preprocess for VGG19.
        """
        img = Image.open(path)
        img = img.resize((512, 512))  #Resize for faster processing
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return tf.Variable(tf.convert_to_tensor(img), dtype=tf.float32)  #Ensure it's a tensor for training

    def convert_content_img(self):
        return self.load_and_process_image(self.content_img_path)

    def convert_style_img(self):
        return self.load_and_process_image(self.style_img_path)

    def build_vgg_model(self):
        """
        Load a VGG19 model trained on ImageNet and extract feature layers.
        """
        vgg = VGG19(weights="imagenet", include_top=False)
        vgg.trainable = False  # Freeze the model
        outputs = {layer.name: layer.output for layer in vgg.layers}
        return Model(inputs=vgg.input, outputs=outputs)

    def get_features(self, image):
        """
        Extract content and style features from an image.
        """
        features = self.model(image)
        content_features = features[self.content_layer]
        style_features = [features[layer] for layer in self.style_layers]
        return content_features, style_features

    def compute_content_loss(self, base_content, target):
        """
        Compute content loss using Mean Squared Error (MSE).
        """
        return tf.reduce_mean(tf.square(base_content - target))

    def gram_matrix(self, tensor):
        """
        Compute Gram matrix for style loss calculation.
        """
        channels = int(tensor.shape[-1])
        vectorized = tf.reshape(tensor, [-1, channels])
        gram = tf.matmul(tf.transpose(vectorized), vectorized)
        return gram

    def compute_style_loss(self, base_style, target_style):
        """
        Compute style loss using Gram matrices.
        """
        base_gram = self.gram_matrix(base_style)
        target_gram = self.gram_matrix(target_style)
        return tf.reduce_mean(tf.square(base_gram - target_gram))

    @tf.function
    def train_step(self, generated_image, content_features, style_features, optimizer):
        """
        Perform a single step of training to adjust the generated image.
        """
        with tf.GradientTape() as tape:
            gen_content_features, gen_style_features = self.get_features(generated_image)

            #Compute content and style loss
            content_loss = self.compute_content_loss(gen_content_features, content_features)
            style_loss = sum(
                [self.compute_style_loss(gen_style_features[i], style_features[i]) for i in range(len(self.style_layers))]
            )

            total_loss = content_loss * 1.0 + style_loss * 100.0  #Adjust weights as needed

        gradients = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])

        return total_loss

    def optimization_loop(self):
        """
        Runs optimization but ensures it exits after a fixed number of iterations or if loss stops improving.
        """
        content_img = self.convert_content_img()
        style_img = self.convert_style_img()

        content_features, _ = self.get_features(content_img)
        _, style_features = self.get_features(style_img)

        #Initialize generated image as content image
        generated_image = tf.Variable(content_img, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=5.0)

        epochs = 1000  #Maximum iterations
        min_loss_delta = 1e-4  #Stop if loss doesn't improve by this much
        last_loss = float("inf")  #Track last loss for early stopping

        print("Optimization process started...\n")

        for i in range(epochs):
            loss = self.train_step(generated_image, content_features, style_features, optimizer)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss.numpy()}")

            #Early stopping condition
            if abs(last_loss - loss.numpy()) < min_loss_delta:
                print(f"Stopping early at iteration {i}: Loss = {loss.numpy()}")
                break 

            last_loss = loss.numpy() #Update last loss 

        print("Optimization completed!")

        #Convert back to an image and save
        final_img = generated_image.numpy().squeeze()
        final_img = np.clip(final_img, 0, 255).astype("uint8")
        Image.fromarray(final_img).save("output_painting.jpg")
        print("Final image saved as 'output_painting.jpg'")


ai_img_generator = ImageGenerator()