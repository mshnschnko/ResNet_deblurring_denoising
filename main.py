from cProfile import label
from sol5 import *
from PIL import Image
from tensorflow.keras import preprocessing
import tifffile as tff

def res_visual(history):
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    # plt.plot(epochs, acc, 'b', label = 'Training acc')
    # plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    img = read_image("low_res/bead_00.tif", 1)
    model, hist = learn_denoising_model(quick_mode = True)
    res_img = restore_image(img, model)
    # tff.imsave("testRTFF.tif", np.reshape(res_img, [36, 36, 1]))
    preprocessing.image.save_img("testR2.tif", np.reshape(res_img, [36, 36, 1]))
    res_visual(hist)
    # im = Image.fromarray(res_img.astype('uint8'))
    
    # # im.show()
    # im = im.convert("L")
    # im.save("testR.jpeg")

# if __name__ == "__main__":
#     img = read_image("low_res/img24wb.jpg", 1)
#     print(type(img))
#     res_img = restore_image(img, learn_deblurring_model())
#     im = Image.fromarray(res_img)
#     im.show()
#     im = im.convert("L")
#     im.save("img24wbR.jpg")
#     # res_img.show()