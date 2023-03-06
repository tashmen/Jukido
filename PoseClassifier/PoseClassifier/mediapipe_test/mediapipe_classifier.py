from mediapipe_common import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#classes = ['', 'dai-kokutsu', 'fudo', 'gankaku', 'hangetsu', 'kake', 'kiba', 'kokutsu', 'musubi', 'neko-ashi', 'sanchin', 'seiko', 'seoi-otoshi', 'tsuru-ashi', 'zenkutsu']
#classes = ['zenkutsu']
#classes = ['fudo', 'seiko', 'sanchin', 'zenkutsu']
classes = ['']
showPicture = True

for binaryClassName in classes:

    isBinary = binaryClassName != ''

    # Output folders for bootstrapped images and CSVs.
    bootstrap_images_out_folder = 'E:/Jukido/Fun/Tensorflow/PythonScripts/jukido_stances_out'
    bootstrap_csvs_out_folder = bootstrap_images_out_folder

    # Transforms pose landmarks into embedding.
    pose_embedder = FullBodyPoseEmbedder()

    # loads the poses
    pose_loader = PoseLoader(pose_samples_folder=bootstrap_csvs_out_folder + '/train', binary_class_name=binaryClassName)

    inputs = tf.keras.Input(shape=(132))
    embedding = pose_embedder(inputs)
    
    layer = keras.layers.Dense(1024, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(512, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    outputs = keras.layers.Dense(pose_loader.numberOfClasses, activation="softmax")(layer)

    model = keras.Model(inputs, outputs)
    model.summary()

    if isBinary:
        lossFunction = 'binary_crossentropy'
        metrics = ['binary_accuracy']
    else:
        lossFunction = 'categorical_crossentropy' 
        metrics = ['accuracy']
    model.compile(
        optimizer='adam',
        loss= lossFunction,
        metrics=['accuracy'],
        #run_eagerly=True
    )

    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                  patience=20)

    # Start training
    if isBinary:
        class_weight = {0: 1.,
                1: 4.}
    else:
        class_weights = pose_loader.class_weights
    history = model.fit(pose_loader.x_input, pose_loader.y_labels,
                        epochs=200,
                        batch_size=32,
                        class_weight = class_weights,
                        validation_split=0.1,
                        callbacks=[checkpoint, earlystopping])

    # Visualize the training history to see whether you're overfitting.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['TRAIN', 'VAL'], loc='lower right')
    if showPicture:
        plt.show()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    print('Model size: %dKB' % (len(tflite_model) / 1024))

    with open('pose_classifier_' + binaryClassName + '.tflite', 'wb') as f:
      f.write(tflite_model)

    with open('pose_labels_' + binaryClassName + '.txt', 'w') as f:
      f.write('\n'.join(pose_loader.class_names))



    pose_loader_test = PoseLoader(pose_samples_folder=bootstrap_csvs_out_folder + '/test', binary_class_name=binaryClassName);
    # Evaluate the model using the TEST dataset
    loss, accuracy = model.evaluate(pose_loader_test.x_input, pose_loader_test.y_labels)

    # Classify pose in the TEST dataset using the trained model
    y_prediction = model.predict(pose_loader_test.x_input)

    # Convert the prediction result to class name
    y_pred_label = [pose_loader_test.class_names[i] for i in np.argmax(y_prediction, axis=1)]
    y_true_label = [pose_loader_test.class_names[i] for i in np.argmax(pose_loader_test.y_labels, axis=1)]

    # Plot the confusion matrix
    cm = confusion_matrix(np.argmax(pose_loader_test.y_labels, axis=1), np.argmax(y_prediction, axis=1))
    plot_confusion_matrix(cm,
                          pose_loader_test.class_names,
                          title ='Confusion Matrix of Pose Classification Model')

    # Print the classification report
    print('\nClassification Report:\n', classification_report(y_true_label,
                                                              y_pred_label))


  
    IMAGE_PER_ROW = 3
    MAX_NO_OF_IMAGE_TO_PLOT = 30

    # Extract the list of incorrectly predicted poses
    false_predict = [id_in_df for id_in_df in range(len(pose_loader_test.y_labels)) \
                    if y_pred_label[id_in_df] != y_true_label[id_in_df]]
    if len(false_predict) > MAX_NO_OF_IMAGE_TO_PLOT:
        false_predict = false_predict[:MAX_NO_OF_IMAGE_TO_PLOT]

    # Plot the incorrectly predicted images
    row_count = len(false_predict) // IMAGE_PER_ROW + 1
    fig = plt.figure(figsize=(10 * IMAGE_PER_ROW, 10 * row_count))
    for i, id_in_df in enumerate(false_predict):
        ax = fig.add_subplot(row_count, IMAGE_PER_ROW, i + 1)
        image_path = os.path.join(bootstrap_images_out_folder + "/test",
                                    pose_loader_test.getImageName(id_in_df))

        image = cv2.imread(image_path)
        plt.title("Predict: %s; Actual: %s" % (y_pred_label[id_in_df], y_true_label[id_in_df]))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if showPicture:
        plt.show()