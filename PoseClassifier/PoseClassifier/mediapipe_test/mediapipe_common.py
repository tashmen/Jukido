import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
import tqdm
import csv
import tensorflow as tf
import itertools

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

class BootstrapHelper(object):
  """Helps to bootstrap images and filter pose samples for classification."""

  def __init__(self,
               images_in_folder,
               images_out_folder,
               csvs_out_folder):
    self._images_in_folder = images_in_folder
    self._images_out_folder = images_out_folder
    self._csvs_out_folder = csvs_out_folder

    # Get list of pose classes and print image statistics.
    self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.__contains__('.')])
    
  def bootstrap(self, per_pose_class_limit=None):
    """Bootstraps images in a given folder.
    
    Required image in folder (same use for image out folder):
      pushups_up/
        image_001.jpg
        image_002.jpg
        ...
      pushups_down/
        image_001.jpg
        image_002.jpg
        ...
      ...

    Produced CSVs out folder:
      pushups_up.csv
      pushups_down.csv

    Produced CSV structure with pose 3D landmarks:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
    """
    # Create output folder for CVSs.
    if not os.path.exists(self._csvs_out_folder):
      os.makedirs(self._csvs_out_folder)

    for pose_class_name in self._pose_class_names:
      print('Bootstrapping ', pose_class_name, file=sys.stderr)

      # Paths for the pose class.
      images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')
      if not os.path.exists(images_out_folder):
        os.makedirs(images_out_folder)

      with open(csv_out_path, 'w', newline='') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        # Get list of images.
        image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        if per_pose_class_limit is not None:
          image_names = image_names[:per_pose_class_limit]

        # Bootstrap every image.
        for image_name in tqdm.tqdm(image_names):
          # Load image.
          input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
          input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

          # Initialize fresh pose tracker and run it.
          with mp_pose.Pose() as pose_tracker:
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

          # Save image with pose prediction (if pose was detected).
          output_frame = input_frame.copy()
          if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
          output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)
          
          # Save landmarks if pose was detected.
          if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array(
                [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility]
                 for lmk in pose_landmarks.landmark],
                dtype=np.float32)
            assert pose_landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
            csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(np.str).tolist())

          # Draw XZ projection and concatenate with the image.
          projection_xz = self._draw_xz_projection(
              output_frame=output_frame, pose_landmarks=pose_landmarks)
          output_frame = np.concatenate((output_frame, projection_xz), axis=1)

  def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
    img = Image.new('RGB', (frame_width, frame_height), color='white')

    if pose_landmarks is None:
      return np.asarray(img)

    # Scale radius according to the image width.
    r *= frame_width * 0.01

    draw = ImageDraw.Draw(img)
    for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
      # Flip Z and move hips center to the center of the image.
      x1, y1, z1, _ = pose_landmarks[idx_1] * [1, 1, -1, 1] + [0, 0, frame_height * 0.5, 0]
      x2, y2, z2, _ = pose_landmarks[idx_2] * [1, 1, -1, 1] + [0, 0, frame_height * 0.5, 0]

      draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
      draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
      draw.line([x1, z1, x2, z2], width=int(r), fill=color)

    return np.asarray(img)

  def align_images_and_csvs(self, print_removed_items=False):
    """Makes sure that image folders and CSVs have the same sample.
    
    Leaves only intersetion of samples in both image folders and CSVs.
    """
    for pose_class_name in self._pose_class_names:
      # Paths for the pose class.
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

      # Read CSV into memory.
      rows = []
      with open(csv_out_path) as csv_out_file:
        csv_out_reader = csv.reader(csv_out_file, delimiter=',')
        for row in csv_out_reader:
          rows.append(row)

      # Image names left in CSV.
      image_names_in_csv = []

      # Re-write the CSV removing lines without corresponding images.
      with open(csv_out_path, 'w', newline='') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
          if len(row) == 0:
            continue
          image_name = row[0]
          image_path = os.path.join(images_out_folder, image_name)
          if os.path.exists(image_path):
            image_names_in_csv.append(image_name)
            csv_out_writer.writerow(row)
          elif print_removed_items:
            print('Removed image from CSV: ', image_path)

      # Remove images without corresponding line in CSV.
      for image_name in os.listdir(images_out_folder):
        if image_name not in image_names_in_csv:
          image_path = os.path.join(images_out_folder, image_name)
          os.remove(image_path)
          if print_removed_items:
            print('Removed image from folder: ', image_path)

  def print_images_in_statistics(self):
    """Prints statistics from the input image folder."""
    self._print_images_statistics(self._images_in_folder, self._pose_class_names)

  def print_images_out_statistics(self):
    """Prints statistics from the output image folder."""
    self._print_images_statistics(self._images_out_folder, self._pose_class_names)

  def _print_images_statistics(self, images_folder, pose_class_names):
    print('Number of images per pose class:')
    for pose_class_name in pose_class_names:
      n_images = len([
          n for n in os.listdir(os.path.join(images_folder, pose_class_name))
          if not n.startswith('.')])
      print('  {}: {}'.format(pose_class_name, n_images))


import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests

class PoseClassificationVisualizer(object):
  """Keeps track of claassifcations for every frame and renders them."""

  def __init__(self,
               class_name,
               plot_location_x=0.05,
               plot_location_y=0.05,
               plot_max_width=0.4,
               plot_max_height=0.4,
               plot_figsize=(9, 4),
               plot_x_max=None,
               plot_y_max=None,
               counter_location_x=0.85,
               counter_location_y=0.05,
               counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
               counter_font_color='red',
               counter_font_size=0.15):
    self._class_name = class_name
    self._plot_location_x = plot_location_x
    self._plot_location_y = plot_location_y
    self._plot_max_width = plot_max_width
    self._plot_max_height = plot_max_height
    self._plot_figsize = plot_figsize
    self._plot_x_max = plot_x_max
    self._plot_y_max = plot_y_max
    self._counter_location_x = counter_location_x
    self._counter_location_y = counter_location_y
    self._counter_font_path = counter_font_path
    self._counter_font_color = counter_font_color
    self._counter_font_size = counter_font_size

    self._counter_font = None

    self._pose_classification_history = []
    self._pose_classification_filtered_history = []

  def __call__(self,
               frame,
               pose_classification,
               pose_classification_filtered,
               repetitions_count):
    """Renders pose classifcation and counter until given frame."""
    # Extend classification history.
    self._pose_classification_history.append(pose_classification)
    self._pose_classification_filtered_history.append(pose_classification_filtered)

    # Output frame with classification plot and counter.
    output_img = Image.fromarray(frame)

    output_width = output_img.size[0]
    output_height = output_img.size[1]

    # Draw the plot.
    img = self._plot_classification_history(output_width, output_height)
    img.thumbnail((int(output_width * self._plot_max_width),
                   int(output_height * self._plot_max_height)),
                  Image.ANTIALIAS)
    output_img.paste(img,
                     (int(output_width * self._plot_location_x),
                      int(output_height * self._plot_location_y)))

    # Draw the count.
    output_img_draw = ImageDraw.Draw(output_img)
    if self._counter_font is None:
      font_size = int(output_height * self._counter_font_size)
      font_request = requests.get(self._counter_font_path, allow_redirects=True)
      self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)
    output_img_draw.text((output_width * self._counter_location_x,
                          output_height * self._counter_location_y),
                         str(repetitions_count),
                         font=self._counter_font,
                         fill=self._counter_font_color)

    return output_img

  def _plot_classification_history(self, output_width, output_height):
    fig = plt.figure(figsize=self._plot_figsize)

    for classification_history in [self._pose_classification_history,
                                   self._pose_classification_filtered_history]:
      y = []
      for classification in classification_history:
        if classification is None:
          y.append(None)
        elif self._class_name in classification:
          y.append(classification[self._class_name])
        else:
          y.append(0)
      plt.plot(y, linewidth=7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.title('Classification history for `{}`'.format(self._class_name))
    plt.legend(loc='upper right')

    if self._plot_y_max is not None:
      plt.ylim(top=self._plot_y_max)
    if self._plot_x_max is not None:
      plt.xlim(right=self._plot_x_max)

    # Convert plot to image.
    buf = io.BytesIO()
    dpi = min(
        output_width * self._plot_max_width / float(self._plot_figsize[0]),
        output_height * self._plot_max_height / float(self._plot_figsize[1]))
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img

from tensorflow import keras

class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
    self._landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding
    
    Args:
      landmarks - Keras input of size 132

    Result:
      Numpy array with pose embedding of shape (33, 4).
    """

    # Get embedding.
    embedding = self.landmarks_to_embedding(landmarks)

    return embedding

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""

    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    pose_center = tf.expand_dims(pose_center, axis=1)

    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (33 * 3), 33, 3])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = self.get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    center = self.get_center_point(landmarks, self._landmark_names.index('left_hip'), self._landmark_names.index('right_hip'))
    return center

  def get_center_point(self, landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""

    left = tf.gather(landmarks, left_bodypart, axis=1)
    right = tf.gather(landmarks, right_bodypart, axis=1)
    center = left * 0.5 + right * 0.5
    return center

  def get_pose_size(self, landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.
    
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """

    # Hips center.
    hips_center = self.get_center_point(landmarks, self._landmark_names.index('left_hip'), 
                                 self._landmark_names.index('right_hip'))

    # Shoulders center.
    shoulders_center = self.get_center_point(landmarks, self._landmark_names.index('left_shoulder'),
                                      self._landmark_names.index('right_shoulder'))

    # Pose center
    pose_center_new = self.get_center_point(landmarks, self._landmark_names.index('left_hip'), 
                                 self._landmark_names.index('right_hip'))
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)

    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (33*3), 33, 3])

    # Torso size as the minimum body size.
    torso_size = tf.linalg.norm(shoulders_center - hips_center)

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size

  def landmarks_to_embedding(self, landmarks_and_visibility):
    """Converts the input landmarks into a pose embedding."""
    reshaped_inputs = keras.layers.Reshape((33, 4))(landmarks_and_visibility)

    landmarks = self._normalize_pose_landmarks(reshaped_inputs[:, :, :3])
    landmarks = tf.keras.layers.Concatenate(axis=2)([landmarks, reshaped_inputs[:, :, 3:4]])

    # Flatten the normalized landmark coordinates into a vector
    # only use the lower body points.
    tf.keras.layers.Flatten
    embedding = keras.layers.Flatten()(landmarks[:,23:,:])

    points = reshaped_inputs[:, :, :3]
    normalizedPoints = landmarks[:, :, :3]
    leftHip = tf.gather(points, self._landmark_names.index('left_hip'), axis=1)
    leftKnee = tf.gather(points, self._landmark_names.index('left_knee'), axis=1)
    leftAnkle = tf.gather(points, self._landmark_names.index('left_ankle'), axis=1) #reshaped_inputs[:,:26:27,:3]
    leftHeel = tf.gather(points, self._landmark_names.index('left_heel'), axis=1) #reshaped_inputs[:,:28:29,:3]
    leftFoot = tf.gather(points, self._landmark_names.index('left_foot_index'), axis=1) #reshaped_inputs[:,:30:31,:3]

    leftFootAngle1 = self.compute_angle_between_3d_points(leftAnkle, leftHeel, leftFoot)
    #leftFootAngle1 = tf.keras.backend.print_tensor(leftFootAngle1, "leftFootAngle1=")
    leftFootAngle2 = self.compute_angle_between_3d_points(leftHeel, leftFoot, leftAnkle)
    leftFootAngle3 = self.compute_angle_between_3d_points(leftFoot, leftAnkle, leftHeel)

    leftKneeAngle1 = self.compute_angle_between_3d_points(leftKnee, leftAnkle, leftHip)
    leftKneeAngle2 = self.compute_angle_between_3d_points(leftAnkle, leftHip, leftKnee)
    leftKneeAngle3 = self.compute_angle_between_3d_points(leftHip, leftKnee, leftAnkle)

    rightHip = tf.gather(points, self._landmark_names.index('right_hip'), axis=1)
    rightKnee = tf.gather(points, self._landmark_names.index('right_knee'), axis=1)
    rightAnkle = tf.gather(points, self._landmark_names.index('right_ankle'), axis=1) 
    rightHeel = tf.gather(points, self._landmark_names.index('right_heel'), axis=1) 
    rightFoot = tf.gather(points, self._landmark_names.index('right_foot_index'), axis=1) 

    rightFootAngle1 = self.compute_angle_between_3d_points(rightAnkle, rightHeel, rightFoot)
    rightFootAngle2 = self.compute_angle_between_3d_points(rightHeel, rightFoot, rightAnkle)
    rightFootAngle3 = self.compute_angle_between_3d_points(rightFoot, rightAnkle, rightHeel)

    rightKneeAngle1 = self.compute_angle_between_3d_points(rightKnee, rightAnkle, rightHip)
    rightKneeAngle2 = self.compute_angle_between_3d_points(rightAnkle, rightHip, rightKnee)
    rightKneeAngle3 = self.compute_angle_between_3d_points(rightHip, rightKnee, rightAnkle)

    
    leftKnee = tf.gather(normalizedPoints, self._landmark_names.index('left_knee'), axis=1)
    rightKnee = tf.gather(normalizedPoints, self._landmark_names.index('right_knee'), axis=1)
    leftFoot = tf.gather(normalizedPoints, self._landmark_names.index('left_foot_index'), axis=1)
    rightFoot = tf.gather(normalizedPoints, self._landmark_names.index('right_foot_index'), axis=1) 
    kneeDistance = self.compute_distance(leftKnee, rightKnee)
    feetDistance = self.compute_distance(leftFoot, rightFoot)
    #kneeDistance = tf.keras.backend.print_tensor(kneeDistance, "KneeDistanceNorm:")

    embedding = tf.keras.layers.Concatenate(axis=1)([embedding, feetDistance, kneeDistance, leftFootAngle1, leftFootAngle2, leftFootAngle3, rightFootAngle1, rightFootAngle2, rightFootAngle3, leftKneeAngle1, leftKneeAngle2, leftKneeAngle3, rightKneeAngle1, rightKneeAngle2, rightKneeAngle3])
    #embedding = tf.keras.layers.Concatenate(axis=1)([embedding, leftFootAngle1])
    #embedding = keras.layers.Flatten()(embedding)
    return embedding
  def compute_distance(self, a, b):
    return tf.linalg.norm((a - b), axis=1, keepdims=True)

  def compute_angle_between_3d_points(self,a,b,c):
    #a = tf.keras.backend.print_tensor(a, "a=")
    #b = tf.keras.backend.print_tensor(b, "b=")
    #c = tf.keras.backend.print_tensor(c, "c=")
    ba = a - b
    #ba = tf.keras.backend.print_tensor(ba, "ba=")
    bc = c - b
    #bc = tf.keras.backend.print_tensor(bc, "bc=")
    babc = ba*bc
    #babc = tf.keras.backend.print_tensor(babc, "babc=")

    cosine_numerator = tf.math.reduce_sum(babc, axis=1, keepdims=True)
    #cosine_numerator = tf.keras.backend.print_tensor(cosine_numerator, "cosine_numerator=")
    cosine_denominator_1 = tf.linalg.norm(ba, axis=1, keepdims=True)
    #cosine_denominator_1 = tf.keras.backend.print_tensor(cosine_denominator_1, "cosine_denominator_1=")
    cosine_denominator_2 = tf.linalg.norm(bc, axis=1, keepdims=True)
    #cosine_denominator_2 = tf.keras.backend.print_tensor(cosine_denominator_2, "cosine_denominator_2=")
    cosine_angle = tf.math.divide_no_nan(cosine_numerator, (cosine_denominator_1 * cosine_denominator_2))
    #cosine_angle = tf.keras.backend.print_tensor(cosine_angle, "cosine_angle=")
    angles = self.acosTF(cosine_angle) / 0.017453292519943295
    angles = angles / 180

    return angles

  def acosTF(self, x, margin=1e-5):
    """ Approximate arccos() as it's not supported within TFLite
    """
    x = tf.clip_by_value(x, margin - 1., 1. - margin)

    # set initial approximation
    xp = tf.abs(x)
    t = tf.sqrt(1. - xp)

    # fix with polynomial
    c3 = -0.0200752
    c2 = xp * c3 + 0.0759031
    c1 = xp * c2 - 0.2126757
    c0 = xp * c1 + 1.5707963
    p = t * c0

    # correct for negative argument
    TAU = 2 * np.pi
    n = TAU / 2. - p
    y = tf.where(x >= 0., p, n)

    return y


class EMADictSmoothing(object):
  """Smoothes pose classification."""

  def __init__(self, window_size=10, alpha=0.2):
    self._window_size = window_size
    self._alpha = alpha

    self._data_in_window = []

  def __call__(self, data):
    """Smoothes given pose classification.

    Smoothing is done by computing Exponential Moving Average for every pose
    class observed in the given time window. Missed pose classes arre replaced
    with 0.
    
    Args:
      data: Dictionary with pose classification. Sample:
          {
            'pushups_down': 8,
            'pushups_up': 2,
          }

    Result:
      Dictionary in the same format but with smoothed and float instead of
      integer values. Sample:
        {
          'pushups_down': 8.3,
          'pushups_up': 1.7,
        }
    """
    # Add new data to the beginning of the window for simpler code.
    self._data_in_window.insert(0, data)
    self._data_in_window = self._data_in_window[:self._window_size]

    # Get all keys.
    keys = set([key for data in self._data_in_window for key, _ in data.items()])

    # Get smoothed values.
    smoothed_data = dict()
    for key in keys:
      factor = 1.0
      top_sum = 0.0
      bottom_sum = 0.0
      for data in self._data_in_window:
        value = data[key] if key in data else 0.0

        top_sum += factor * value
        bottom_sum += factor

        # Update factor.
        factor *= (1.0 - self._alpha)

      smoothed_data[key] = top_sum / bottom_sum

    return smoothed_data

class PoseSample(object):

  def __init__(self, name, landmarks, class_name):
    self.name = name
    self.landmarks = landmarks
    self.class_name = class_name


class PoseSampleOutlier(object):

  def __init__(self, sample, detected_class, all_classes):
    self.sample = sample
    self.detected_class = detected_class
    self.all_classes = all_classes

import csv
import numpy as np
import os

class PoseLoader(object):
  """Classifies pose landmarks."""

  def __init__(self,
               pose_samples_folder,
               file_extension='csv',
               file_separator=',',
               n_landmarks=33,
               n_dimensions=4,
               binary_class_name = ''):
    self._n_landmarks = n_landmarks
    self._n_dimensions = n_dimensions


    self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                 file_extension,
                                                 file_separator,
                                                 n_landmarks,
                                                 n_dimensions)

    data = []
    classSamples = []
    sampleCount = []
    className = ''
    self.class_names = []
    y = []

    if binary_class_name == '':
        for sample in self._pose_samples:
            if className == '' or className != sample.class_name:
                if len(classSamples) > 0:
                    self.class_names.append(className)
                    data.append(classSamples)
                    sampleCount.append(len(classSamples))
                    classSamples = []
                className = sample.class_name

            classSamples.append(np.transpose(sample.landmarks));

        self.class_names.append(className)
        sampleCount.append(len(classSamples));
        data.append(classSamples)

        i = 0
        for count in sampleCount:
            y.append(np.full(count, i))
            i=i+1

        self.numberOfClasses = len(data)
        self.y_labels = keras.utils.to_categorical(np.concatenate(y));
        self.x_input = np.concatenate(data)
    else :
        self.class_names = [binary_class_name, 'not ' + binary_class_name]
        for sample in self._pose_samples:
            data.append([np.transpose(sample.landmarks)])
            if binary_class_name == sample.class_name:
                y.append(0)
            else :
                y.append(1)
        self.numberOfClasses = 2
        self.y_labels = keras.utils.to_categorical(y)
        self.x_input = np.concatenate(data)
    

  def _load_pose_samples(self,
                         pose_samples_folder,
                         file_extension,
                         file_separator,
                         n_landmarks,
                         n_dimensions):
    """Loads pose samples from a given folder.
    
    Required folder structure:
      neutral_standing.csv
      pushups_down.csv
      pushups_up.csv
      squats_down.csv
      ...

    Required CSV structure:
      sample_00001,x1,y1,z1,v1,x2,y2,z2,v2,....
      sample_00002,x1,y1,z1,v1,x2,y2,z2,v2,....
      ...
    """
    # Each file in the folder represents one pose class.
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    pose_samples = []
    for file_name in file_names:
      # Use file name as pose class name.
      class_name = file_name[:-(len(file_extension) + 1)]
      
      # Parse CSV.
      with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
          assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
          landmarks = np.array(row[1:], np.float32)
          pose_samples.append(PoseSample(
              name=class_name + "/" + row[0],
              landmarks=landmarks,
              class_name=class_name
          ))

    return pose_samples


from matplotlib import pyplot as plt


def show_image(img, figsize=(10, 10)):
  """Shows output PIL image."""
  plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.show()




import os
import random
import shutil

def split_into_train_test(images_origin, images_dest, test_split):
  """Splits a directory of sorted images into training and test sets.

  Args:
    images_origin: Path to the directory with your images. This directory
      must include subdirectories for each of your labeled classes. For example:
      yoga_poses/
      |__ downdog/
          |______ 00000128.jpg
          |______ 00000181.jpg
          |______ ...
      |__ goddess/
          |______ 00000243.jpg
          |______ 00000306.jpg
          |______ ...
      ...
    images_dest: Path to a directory where you want the split dataset to be
      saved. The results looks like this:
      split_yoga_poses/
      |__ train/
          |__ downdog/
              |______ 00000128.jpg
              |______ ...
      |__ test/
          |__ downdog/
              |______ 00000181.jpg
              |______ ...
    test_split: Fraction of data to reserve for test (float between 0 and 1).
  """
  _, dirs, _ = next(os.walk(images_origin))

  TRAIN_DIR = os.path.join(images_dest, 'train')
  TEST_DIR = os.path.join(images_dest, 'test')
  os.makedirs(TRAIN_DIR, exist_ok=True)
  os.makedirs(TEST_DIR, exist_ok=True)

  for dir in dirs:
    # Get all filenames for this dir, filtered by filetype
    filenames = os.listdir(os.path.join(images_origin, dir))
    filenames = [os.path.join(images_origin, dir, f) for f in filenames if (
        f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.bmp'))]
    # Shuffle the files, deterministically
    filenames.sort()
    random.seed(42)
    random.shuffle(filenames)
    # Divide them into train/test dirs
    os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)
    test_count = int(len(filenames) * test_split)
    for i, file in enumerate(filenames):
      if i < test_count:
        destination = os.path.join(TEST_DIR, dir, os.path.split(file)[1])
      else:
        destination = os.path.join(TRAIN_DIR, dir, os.path.split(file)[1])
      shutil.copyfile(file, destination)
    print(f'Moved {test_count} of {len(filenames)} from class "{dir}" into test.')
  print(f'Your split dataset is in "{images_dest}"')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """Plots the confusion matrix."""
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=55)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()