import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def fourier_compute(matrix):
    # Aplicar la transformada rápida de Fourier.
    m_fft = np.fft.fft2(matrix)

    # Usar fftshift para centrar las bajas frecuencias
    m_fft_shift = np.fft.fftshift(m_fft)

    # Obteniendo el valor absoluto de la transformada
    m_fft_shift_abs = np.abs(m_fft_shift)

    # Calcular el espectro de magnitudes en escala logaritmica
    magnitude_log = np.log(m_fft_shift_abs + 1)
    #magnitude_log_ddB = 20 * magnitude_log

    # Normalizar la imagen del espectro para visualizarla adecuadamente
    magnitud_log_norm = np.uint8(cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX))

    # Devuelve la transformada rápida de fourier (con shift) y el espectro normalizado
    return m_fft, m_fft_shift, magnitud_log_norm

if __name__ == "__main__":

    """
    Transformada de la imagen
    """
    # Cargar la imagen y convertirla a escala de grises
    img = cv2.imread('gojo.jpg', cv2.IMREAD_GRAYSCALE)

    img_fft, img_fft_shift, img_magnitude_spectrum = fourier_compute(img)

    """
    Transformada del filtro
    """
    # Máscara del filtro de bloque de tamaño n
    n = 21
    filter = (1/n**2) * np.ones((n,n))

    # Padding (solamente) del filtro
    height, width = img.shape
    pad_fil_num_zeros_hor = (width - len(filter))//2
    pad_fil_num_zeros_ver = (height - len(filter))//2
    filter_pad = cv2.copyMakeBorder(filter,
                                    pad_fil_num_zeros_ver,
                                    pad_fil_num_zeros_ver + 1,
                                    pad_fil_num_zeros_hor,
                                    pad_fil_num_zeros_hor + 1,
                                    cv2.BORDER_CONSTANT,
                                    value=0)

    filter_fft, filter_fft_shift, filter_magnitude_spectrum = fourier_compute(filter_pad)

    """
    Aplicación de filtro con convolución lineal
    """
    # Aplicar el filtro a la imagen
    img_filtered_lineal = convolve2d(img, filter, mode='same')
    #img_filtered_lineal = cv2.filter2D(img, -1, filter)

    """
    Aplicación del filtro con Fourier
    """
    print(f"Tamaño imagen: {height}")
    print(f"Tamaño filtro con padding: {len(filter_pad)}")

    # Multiplicación de imagen y filtro
    img_filtered_fft = img_fft_shift * filter_fft_shift

    # Transformada inversa
    img_filtered_ifft = np.fft.ifft2(img_filtered_fft)

    # Shift inverso
    img_filtered_ifft = np.fft.ifftshift(img_filtered_ifft)

    # Obteniendo el valor absoluto de la imagen filtrada
    img_filtered_ifft_abs = np.array(np.abs(img_filtered_ifft),dtype=np.uint8)

    new_fft, new_fft_shift, new_magnitude_spectrum = fourier_compute(img_filtered_ifft_abs)

    """
    Padding a image y filtro
    """
    new_size = height + len(filter) - 1

    # Padding a imagen
    pad_img_num_zeros = (new_size - height)//2
    img_pad = cv2.copyMakeBorder(img,
                                 pad_img_num_zeros,
                                 pad_img_num_zeros,
                                 pad_img_num_zeros,
                                 pad_img_num_zeros,
                                 cv2.BORDER_CONSTANT,
                                 value=0)

    # Nuevo padding a filtro
    new_pad_fil_num_zeros = (new_size - len(filter)) // 2
    new_filter_pad = cv2.copyMakeBorder(filter,
                                        new_pad_fil_num_zeros,
                                        new_pad_fil_num_zeros + 1,
                                        new_pad_fil_num_zeros,
                                        new_pad_fil_num_zeros + 1,
                                        cv2.BORDER_CONSTANT,
                                        value=0)

    print(f"Tamaño de nueva imagen para padding: {new_size}")
    print(f"Tamaño imagen con padding: {len(img_pad)}")
    print(f"Tamaño filtro con padding: {len(new_filter_pad)}")

    # Obrención de las transformadas para las nuevas imagenes con padding
    pad_img_fft, pad_img_fft_shift, pad_img_magnitude_spectrum = fourier_compute(img_pad)
    new_filter_pad_fft, new_filter_pad_fft_shift, new_filter_pad_magnitude_spectrum = fourier_compute(new_filter_pad)

    # Multiplicación de imagen y filtro
    pad_img_filtered_fft = pad_img_fft_shift * new_filter_pad_fft_shift

    # Transformada inversa
    pad_img_filtered_ifft = np.fft.ifft2(pad_img_filtered_fft)

    # Shift inverso
    pad_img_filtered_ifft = np.fft.ifftshift(pad_img_filtered_ifft)

    # Obteniendo el valor absoluto de la imagen filtrada
    pad_img_filtered_ifft_abs = np.array(np.abs(pad_img_filtered_ifft),dtype=np.uint8)

    pad_fft, pad_fft_shift, pad_magnitude_spectrum = fourier_compute(img_filtered_ifft_abs)

    """
    Guardado de imágenes
    """
    cv2.imwrite('imagen_original.png', img)
    cv2.imwrite('espectro_fourier_original.png', img_magnitude_spectrum)
    cv2.imwrite('espectro_fourier_filtro.png', filter_magnitude_spectrum)
    cv2.imwrite('imagen_fil_fourier_pad_filtro.png', img_filtered_ifft_abs)
    cv2.imwrite('imagen_fil_lineal.png', img_filtered_lineal)
    cv2.imwrite('espectro_fourier_img_filtrada.png', new_magnitude_spectrum)
    cv2.imwrite('imagen_fil_fourier_pad_ambas.png', pad_img_filtered_ifft_abs)

    """
    MOstrado de imágenes
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(filter_pad, cmap='gray')
    plt.title("Filtro")
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(img_filtered_ifft_abs, cmap='gray')
    plt.title('Fourier (padding solo en filtro)')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(pad_img_filtered_ifft_abs, cmap='gray')
    plt.title('Fourier (padding en filtro e imagen)')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.imshow(img_magnitude_spectrum)
    plt.title('Espectro de Fourier de imagen')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(filter_magnitude_spectrum)
    plt.title("Espectro de Fourier Filtro")
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(new_magnitude_spectrum)
    plt.title('Espectro de la imagen filtrada fourier')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(img_filtered_lineal, cmap='gray')
    plt.title('Imagen Filtrada linealmente')
    plt.axis('off')

    plt.tight_layout()
    plt.show()