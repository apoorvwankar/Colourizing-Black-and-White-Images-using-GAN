import streamlit as st
import tensorflow as tf
from PIL import Image
import io

def get_generator_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=1, padding='same')(inputs)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, padding='same')(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, padding='same')(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)

    conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=1, padding='same')(conv1)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(conv2)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same')(conv2)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)

    conv3 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, padding='same')(conv2)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)

    bottleneck = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='tanh', padding='same')(conv3)

    concat_1 = tf.keras.layers.Concatenate()([bottleneck, conv3])
    conv_up_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(concat_1)
    conv_up_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(conv_up_3)
    conv_up_3 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=1, activation='relu', padding='same')(conv_up_3)

    concat_2 = tf.keras.layers.Concatenate()([conv_up_3, conv2])
    conv_up_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(concat_2)
    conv_up_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(conv_up_2)
    conv_up_2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=1, activation='relu', padding='same')(conv_up_2)

    concat_3 = tf.keras.layers.Concatenate()([conv_up_2, conv1])
    conv_up_1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(concat_3)
    conv_up_1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(conv_up_1)
    conv_up_1 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=1, activation='relu', padding='same')(conv_up_1)

    model = tf.keras.models.Model(inputs, conv_up_1)
    return model

@st.cache(allow_output_mutation=True)
def colorize_image(image, generator):
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    colorized_image = generator.predict(tf.expand_dims(image_array, axis=0))[0]
    colorized_image = Image.fromarray((255 * colorized_image).astype('uint8'))
    return colorized_image

def app():
    st.title("Image Colorization App")
    uploaded_file = st.file_uploader("Upload a grayscale image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        input_shape = (*image.size[::-1], 1)

        generator = get_generator_model(input_shape)
        generator.load_weights('generator_weights.h5')

        st.image(image, caption="Grayscale Image", use_column_width=True)

        if st.button("Convert"):
            colorized_image = colorize_image(image, generator)
            st.image(colorized_image, caption="Colorized Image", use_column_width=True)

            img_bytes = io.BytesIO()
            colorized_image.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()
            st.download_button(label="Download Colorized Image", data=img_bytes, file_name="colorized_image.jpg", mime="image/jpeg")

if __name__ == '__main__':
    app()
