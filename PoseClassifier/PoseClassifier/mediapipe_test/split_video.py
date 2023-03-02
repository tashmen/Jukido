import cv2
import os
baseLocation = 'E:\\Jukido\\Fun\\Tensorflow\\PythonScripts\\'
movieLocation = 'Movies\\'
tempLocation = movieLocation + 'temp\\'
imagesLocation = 'jukido_stances\\'

targetLocation = baseLocation + imagesLocation

sourceMovieLocation = baseLocation + tempLocation
for filename in os.listdir(sourceMovieLocation):
    print('processing: ' + filename)
    baseFileName = filename[0:len(filename) - 4]
    splitFileName = baseFileName.split('_')
    folder = splitFileName[1];
    
    sourceMoviePath = os.path.join(sourceMovieLocation, filename)
    processedVideoDirectory = os.path.join(baseLocation + movieLocation, folder)
    

    targetDirectory = os.path.join(baseLocation + imagesLocation, folder)
    if not os.path.exists(targetDirectory):
        os.mkdir(targetDirectory)

    print('target image location: ' + targetDirectory)

    vidcap = cv2.VideoCapture(sourceMoviePath)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in this vid :", length )
    count = 0
    skipframes = 3 # no of frames to skip
    print('started slicing..')
    success = True
    while success:
      success,image = vidcap.read()
      if (count % skipframes == 0):
        if success:
            imagePath = os.path.join(targetDirectory, baseFileName + "_%s.jpg" % str(count).zfill(4))
            cv2.imwrite(imagePath, image)     # save frame as JPEG file
        else:
            break

      count += 1

    vidcap.release()

    print('moving video to: ' + processedVideoDirectory)
    if not os.path.exists(processedVideoDirectory):
        os.mkdir(processedVideoDirectory)
    os.rename(sourceMoviePath, os.path.join(processedVideoDirectory, filename))
  
print('finished..') 
