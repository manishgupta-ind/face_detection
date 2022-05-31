from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN


class face_detection():
    """
    Objective : Take an image file as input and detects face present in image.
    get_face_features(): This method will print on screen as well as return coordinates, 
                         confidence and other features of detected faces in image.
    image_with_faces(): This method will display image with detected faces.
    get_faces(): This method will extract and plot each detected face in image separately. 
                 It will also return extracted faces from image which can be used for further processing.
    """

    def __init__(self, image):   

      # create the detector, using default weights of pretrained model in MTCNN module
      detector = MTCNN() 

      self.pixels = image

      # detect faces in the image
      self.faces = detector.detect_faces(self.pixels)

      if len(self.faces) == 0:
        print("Warning!!! No face detected in input image")
      else:
        print("Total {} faces detected in input image".format(len(self.faces)))

    # print coordinates, confidence and other features of detected faces
    def get_face_features(self):
      # print coordinates, confidence and other features of detected faces
      for face in self.faces:
        print(face)

      return self.faces

    # display image with detected faces
    def image_with_faces(self):
      plt.imshow(self.pixels)
      # get the context for drawing boxes
      ax = plt.gca()
      # plot each box in image
      for face in self.faces:
        x, y, width, height = face['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
      # show the final plot
      plt.show()

    # extract and plot each detected face in image separately
    def get_faces(self):
      detected_faces = []

      # plot each face as a subplot
      for i in range(len(self.faces)):
        x1, y1, width, height = self.faces[i]['box']
        x2, y2 = x1 + width, y1 + height
        plt.subplot(1, len(self.faces), i+1)
        plt.axis('off')
        plt.imshow(self.pixels[y1:y2, x1:x2])
        detected_faces.append(self.pixels[y1:y2, x1:x2])
      # show the plot
      plt.show()
      return detected_faces  