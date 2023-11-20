import numpy as np
import time

image = np.random.random((3,28,28)) #generate 2d image data from [0, 1)

#used for fastfowarding of convolution neural network
class conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if padding == 'same':
            self.padding = 0
    def __call__(self, image):
        if image.shape[0] != self.in_channels:
            print('input channel does not match.')
        else:
            #setting up the dimensions
            image_channel, image_height, image_width = image.shape[0], image.shape[1], image.shape[2]
            #kenerl size
            filter_height, filter_width = self.kernel_size, self.kernel_size
            kernel = np.random.random((1, filter_height, filter_width))
            #output construction
            output_height = int(((image_height - filter_height + 2 * self.padding) / self.stride) + 1)
            output_width = int(((image_width - filter_width + 2 * self.padding) / self.stride) + 1)
            output = np.zeros((self.out_channels, output_height, output_width))
            #padding fuction for the input image 
            if self.padding != 0:
                #padded added
                padded_image = np.zeros((self.in_channels, image_height + 2 * self.padding, image_width + 2 * self.padding ))
                #use negative index to put image in the padded zeros
                padded_image[:, self.padding:(-1 * self.padding), self.padding:(-1 * self.padding)] = image 
            #user report of padded
                print('='*50)
                print('padded_image')
                print(f'padded_image shape : {padded_image.shape}')
                print(padded_image)
            else: 
                padded_image = image
            #cross correlation operation for 2d image 
            for z in range(self.out_channels):
                #setting outputs with zeros first for each filter
                output_per_filter = np.zeros((output_height, output_width))
                #croess correlation operation done with stride applied 
                for y in range(output_height):
                    #each start of the operation starts at y * stride
                    #the final start of the operation which is the size of filter always must be lesser than padded_image_height
                    if (y * self.stride + filter_height) <= padded_image.shape[1]:
                        for x in range(output_width):
                            #each start starts at x * stride 
                            #the final start must be lesser than the padded_image-width
                            if (x * self.stride + filter_width) <= padded_image.shape[2]:
                                #summing of cross correlation operation after multiplying it with filter, 3d multiplication used for multiple filters 
                                output_per_filter[y][x] = np.sum(padded_image[:, y * self.stride: y * self.stride + filter_height, x * self.stride : x * self.stride + filter_width] * kernel)
                output[z, :, :] = output_per_filter
            #user reporting of output
            print('='*50)
            print(output)
            print('='*50)
            print(f'output shape : {output.shape}')
            return output

cnn = conv2d(3, 32, 1, 1, padding='same')

start = time.time()
print(cnn(image).shape)
end = time.time()

print(end - start)