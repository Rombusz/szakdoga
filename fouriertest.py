import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

global magnitude_spectrum
global magnitude_spectrum2

global slider_band
global slider_offset

slider_offset = 0
slider_band = 0

magnitude_spectrum=0
magnitude_spectrum2=0

def onchoffset(val):

    global magnitude_spectrum
    global magnitude_spectrum2

    global slider_band
    global slider_offset

    slider_offset = val
    mask = np.zeros((img.shape[0], img.shape[1], 2))
    mask2 = np.zeros((img2.shape[0], img2.shape[1], 2))

    m_offset = int( mask.shape[1]*slider_offset )
    m_offset2 = int(  mask2.shape[1]*slider_offset )

    m_band = int(  mask.shape[1]*slider_band )
    m_band2 = int(  mask2.shape[1]*slider_band )

    mask[:,m_offset-m_band:m_offset+m_band] = 1
    mask2[:,m_offset2-m_band2:m_offset2+m_band2] = 1

    m_offset = int( mask.shape[0]*slider_offset )
    m_offset2 = int(  mask2.shape[0]*slider_offset )

    m_band = int(  mask.shape[0]*slider_band )
    m_band2 = int(  mask2.shape[0]*slider_band )

    mask[m_offset-m_band:m_offset+m_band,:] = 1
    mask2[m_offset2-m_band2:m_offset2+m_band2,:] = 1

    print("inverse transform started")

    print(m_offset,m_offset2,m_band,m_band2)

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # apply mask and inverse DFT
    fshift2 = dft_shift2 * mask2
    f_ishift2 = np.fft.ifftshift(fshift2)
    img_back2 = cv2.idft(f_ishift2)
    img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

    magnitude_spec = magnitude_spectrum * mask[:, :, 0]
    magnitude_spec2 = magnitude_spectrum2 * mask2[:, :, 0]

    axarr[1, 0].imshow(magnitude_spec, cmap='gray')
    axarr[1, 0].set_title('Magnitude Spectrum original'), axarr[1, 0].set_xticks([]), axarr[1, 0].set_yticks([])

    axarr[1, 1].imshow(magnitude_spec2, cmap='gray')
    axarr[1, 1].set_title('Magnitude Spectrum pyramid'), axarr[1, 1].set_xticks([]), axarr[1, 1].set_yticks([])

    axarr[2, 0].imshow(img_back, cmap='gray')
    axarr[2, 0].set_title('Inverse transform'), axarr[2, 0].set_xticks([]), axarr[2, 0].set_yticks([])

    axarr[2, 1].imshow(img_back2, cmap='gray')
    axarr[2, 1].set_title('Inverse transform pyramid'), axarr[2, 1].set_xticks([]), axarr[2, 1].set_yticks([])

    img_to_write1 = np.interp(img_back,( img_back.min(), img_back.max() ), (0,255))
    img_to_write2 = np.interp(img_back2,( img_back2.min(), img_back2.max() ), (0,255))

    cv2.imwrite("it_orig.png",img_to_write1)
    cv2.imwrite("it_pyr.png",img_to_write2)
    print("inverse transform ended")


def onchband(val):

    global magnitude_spectrum
    global magnitude_spectrum2

    global slider_band
    global slider_offset

    slider_band = val

    mask = np.zeros((img.shape[0], img.shape[1], 2))
    mask2 = np.zeros((img2.shape[0], img2.shape[1], 2))

    m_offset = int( mask.shape[1]*slider_offset )
    m_offset2 = int(  mask2.shape[1]*slider_offset )

    m_band = int(  mask.shape[1]*slider_band )
    m_band2 = int(  mask2.shape[1]*slider_band )

    mask[:,m_offset-m_band:m_offset+m_band] = 1
    mask2[:,m_offset2-m_band2:m_offset2+m_band2] = 1

    m_offset = int( mask.shape[0]*slider_offset )
    m_offset2 = int(  mask2.shape[0]*slider_offset )

    m_band = int(  mask.shape[0]*slider_band )
    m_band2 = int(  mask2.shape[0]*slider_band )

    mask[m_offset-m_band:m_offset+m_band,:] = 1
    mask2[m_offset2-m_band2:m_offset2+m_band2,:] = 1

    print("inverse transform started")
    #print(m_offset, m_offset2, m_ban  d, m_band2)
    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # apply mask and inverse DFT
    fshift2 = dft_shift2 * mask2
    f_ishift2 = np.fft.ifftshift(fshift2)
    img_back2 = cv2.idft(f_ishift2)
    img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

    magnitude_spec = magnitude_spectrum * mask[:, :, 0]
    magnitude_spec2 = magnitude_spectrum2 * mask2[:, :, 0]

    axarr[1, 0].imshow(magnitude_spec, cmap='gray')
    axarr[1, 0].set_title('Magnitude Spectrum original'), axarr[1, 0].set_xticks([]), axarr[1, 0].set_yticks([])

    axarr[1, 1].imshow(magnitude_spec2, cmap='gray')
    axarr[1, 1].set_title('Magnitude Spectrum pyramid'), axarr[1, 1].set_xticks([]), axarr[1, 1].set_yticks([])

    axarr[2, 0].imshow(img_back, cmap='gray')
    axarr[2, 0].set_title('Inverse transform'), axarr[2, 0].set_xticks([]), axarr[2, 0].set_yticks([])

    axarr[2, 1].imshow(img_back2, cmap='gray')
    axarr[2, 1].set_title('Inverse transform pyramid'), axarr[2, 1].set_xticks([]), axarr[2, 1].set_yticks([])

    img_to_write1 = np.interp(img_back,( img_back.min(), img_back.max() ), (0,255))
    img_to_write2 = np.interp(img_back2,( img_back2.min(), img_back2.max() ), (0,255))

    cv2.imwrite("it_orig.png",img_to_write1)
    cv2.imwrite("it_pyr.png",img_to_write2)

    print(img_back)

    print("inverse transform ended")


img = cv2.imread('./images/intensity/kaula_int_4.jpg',0)

img2 = cv2.pyrDown(img)
img2 = cv2.pyrDown(img2)
img2 = cv2.pyrDown(img2)

f, axarr = plt.subplots(3,2)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

dft2 = cv2.dft(np.float32(img2),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift2 = np.fft.fftshift(dft2)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
magnitude_spectrum2 = 20*np.log(cv2.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))

axarr[0,0].imshow(img, cmap = 'gray')
axarr[0,0].set_title('Input Image'), axarr[0,0].set_xticks([]), axarr[0,0].set_yticks([])

axarr[0,1].imshow(img2, cmap = 'gray')
axarr[0,1].set_title('Input Image pyramid'), axarr[0,1].set_xticks([]), axarr[0,1].set_yticks([])


mask = np.ones((img.shape[0], img.shape[1], 2))
mask2 = np.ones((img2.shape[0], img2.shape[1], 2))

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


# apply mask and inverse DFT
fshift2 = dft_shift2*mask2
f_ishift2 = np.fft.ifftshift(fshift2)
img_back2 = cv2.idft(f_ishift2)
img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])

magnitude_spec = magnitude_spectrum*mask[:,:,0]
magnitude_spec2 = magnitude_spectrum2*mask2[:,:,0]

axarr[1,0].imshow(magnitude_spec, cmap = 'gray')
axarr[1,0].set_title('Magnitude Spectrum original'), axarr[1,0].set_xticks([]), axarr[1,0].set_yticks([])

axarr[1,1].imshow(magnitude_spec2, cmap = 'gray')
axarr[1,1].set_title('Magnitude Spectrum pyramid'), axarr[1,1].set_xticks([]), axarr[1,1].set_yticks([])

axarr[2,0].imshow(img_back, cmap = 'gray')
axarr[2,0].set_title('Inverse transform'), axarr[2,0].set_xticks([]), axarr[2,0].set_yticks([])

axarr[2,1].imshow(img_back2, cmap = 'gray')
axarr[2,1].set_title('Inverse transform pyramid'), axarr[2,1].set_xticks([]), axarr[2,1].set_yticks([])

axcolor = 'lightgoldenrodyellow'
axoffs = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)
axband = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)

offset = Slider(axoffs,"offset",0,1,valinit=0.5)
band = Slider(axband,"band",0,0.5,valinit=0.25)

offset.on_changed(onchoffset)
band.on_changed(onchband)

plt.show()