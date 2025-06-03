"""
code here helps to visualise the data
"""
import cv2

def display_image(image_array, window_name="Image", wait_time=0):
    """
    Display an image array using OpenCV

    Args:
        image_array: numpy array representing the image
        window_name: name of the display window
        wait_time: time to wait (0 = wait for key press, >0 = milliseconds)
    """
    # Display the image
    cv2.imshow(window_name, image_array)

    # Wait for key press or specified time
    cv2.waitKey(wait_time)

    # Clean up (optional - removes the window)
    cv2.destroyAllWindows()
