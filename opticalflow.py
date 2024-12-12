import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to count people using optical flow
def count_people(image_sequence):
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(30, 30),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.01,
                          minDistance=50,
                          blockSize=7)
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Iterate through the image sequence
    prev_frame = None
    count = 0
    for frame in image_sequence:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate optical flow
            p0 = cv2.goodFeaturesToTrack(prev_frame, mask=None, **feature_params)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[status == 1]
            good_old = p0[status == 1]

            # Count people based on displacement
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                if abs(a - c) + abs(b - d) > 2:  # Set threshold for movement detection
                    count += 1

        prev_frame = frame_gray.copy()
    marked_image = image_sequence[-1].copy()  # Assuming the last frame is the current frame

    # Draw circles around the detected people
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        if abs(a - c) + abs(b - d) > 2:  # Set threshold for movement detection
            marked_image = cv2.circle(marked_image, (int(a), int(b)), 5, (0, 255, 0), -1)  # Draw a green circle

    return count, marked_image

def read_images_from_folder(folder_path):
    image_sequence = []
    # List all files in the folder
    files = os.listdir(folder_path)
    # Filter out only image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    # Sort the image files alphabetically
    image_files.sort()
    # Read each image file
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        if image is not None:
            image_sequence.append(image)
    return image_sequence

# Example usage
# Load your image sequence into a list
def main():
    for i in range(1,21):
        image_sequence =read_images_from_folder(f"C:/Users/user/Downloads/Crowd_Dataset/Crowd Dataset/Crowd Sequence/Sequence19")
        people_count,img = count_people(image_sequence)
        cv2.imshow('Sequence'+str(i),cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert the image from BGR to RGB for correct display
        #plt.axis('off')  # Turn off axis
        #plt.show()
        print("Sequence "+str(i)+" Number of people: ", people_count)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# Call the function to count people
if __name__ == '__main__':
    main()
