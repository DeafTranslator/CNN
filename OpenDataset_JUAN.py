import tensorflow as tf
import time
from datetime import timedelta
import datetime
import pandas as pd
import random
#=====================
import dataset
import config as conf
import plot as pl
import printStatus as prSt
#=====================
import csv 

data = dataset.read_train_sets(conf.train_path, conf.img_size, conf.classes, validation_size=conf.validation_size)
data_valid = dataset.read_train_validation_sets(conf.validation_path, conf.img_size, conf.classes, validation_size=conf.validation_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data_valid.train.labels)))

# Plot the images and labels using our helper-function above.
#pl.plot_images(images=images, cls_true=cls_true)

graph = tf.Graph()

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape, )

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters, )

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights, biases

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])
    #print("num_feature = {0}\nlayer_shape = {1}\nlayer flat = {2}".format(num_features, layer_shape, layer_flat))
    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    #print(layer)
    return layer, weights, biases

x = tf.placeholder(tf.float32, shape=[None, conf.img_size_flat], name='x') #Crea espacio en memoria de una variable

x_image = tf.reshape(x, [-1, conf.img_size, conf.img_size, conf.num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, conf.num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

#imprimo = "x = {0}, x_image = {1}, y_true = {2}, y_true_cls = {3}"
#print(imprimo.format(x, x_image, y_true, y_true_cls))
with graph.name_scope("layer_Conv1") :
    layer_conv1, weights_conv1, biases_conv1 = \
        new_conv_layer(input=x_image,
                    num_input_channels=conf.num_channels,
                    filter_size=conf.filter_size1,
                    num_filters=conf.num_filters1,
                    use_pooling=True)
with graph.name_scope("layer_Conv2") :
    layer_conv2, weights_conv2, biases_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=conf.num_filters1,
                   filter_size=conf.filter_size2,
                   num_filters=conf.num_filters2,
                   use_pooling=True)

with graph.name_scope("layer_Conv3") :
    layer_conv3, weights_conv3, biases_conv3 = \
        new_conv_layer(input=layer_conv2,
                    num_input_channels=conf.num_filters2,
                    filter_size=conf.filter_size3,
                    num_filters=conf.num_filters3,
                   use_pooling=True)

with graph.name_scope("layer_flat") :
    layer_flat, num_features = flatten_layer(layer_conv3)

with graph.name_scope("layer_fc1") :
    layer_fc1, fc1_weights, fc1_biases = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=conf.fc_size,
                         use_relu=True)

with graph.name_scope("layer_fc2") :
    layer_fc2, fc2_weights, fc2_biases = new_fc_layer(input=layer_fc1,
                         num_inputs=conf.fc_size,
                         num_outputs=conf.num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name="final_result")

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(cost)

correct_prediction  = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = conf.batch_size

# Counter for total number of iterations performed so far.
total_iterations = 0

# image1 = test_images[0]
#pl.plot_image(image1)

def write_predictions(ims, ids):
    ims = ims.reshape(ims.shape[0], conf.img_size_flat)
    preds = session.run(y_pred, feed_dict={x: ims})
    result = pd.DataFrame(preds, columns=conf.classes)
    result.loc[:, 'id'] = pd.Series(ids, index=result.index)
    pred_file = 'predictions.csv'
    result.to_csv(pred_file, index=False)
    print("result = {0}\n".format(result))

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations, total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        # cls_batch name of the class
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data_valid.train.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, conf.img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, conf.img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        
        # Print status at end of each epoch (defined as full pass through training dataset).
        numero = int(data.train.num_examples/conf.batch_size)
        print("iterations = {0}\t patience = {1}".format(i+1, patience))
        if i % int(data.train.num_examples/conf.batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/conf.batch_size))
            
            date = datetime.datetime.fromtimestamp(time.time()).strftime('%c')
            prSt.print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, session, accuracy)
            csvFile = "Epoch" + str(epoch) + "loss: " + str(val_loss) + "accuracy " + str(accuracy) 
            # fd = open("/Users/jesuspereyra/Desktop/log/training/model_18_2017_epoch_" + str(epoch) + str(conf.iterations) + ".csv" , "w")
            # fd.write(csvFile)
            # fd.close()
            # y_pred_cls = tf.argmax(y_pred, dimension=1)
            tf.add_to_collection('y_pred_cls', y_pred_cls)

            #
            tf.add_to_collection('final_result', y_pred)

            name = ('model_epoch' + str(epoch) + 'iteration' + str(i) + '_' + str(val_loss) + ' accuracy: ' + str(accuracy)) 
            # Save model weights to disk
            save_path = saver.save(session, conf.model_path + name)
            tf.train.export_meta_graph(conf.model_path + name)
            #write_predictions(test_images, test_ids)
            print("Model saved in file: %s" % save_path)


            if conf.early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                else:
                    patience += 1

                if patience > 1:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # y_pred_cls = tf.argmax(y_pred, dimension=1)
    tf.add_to_collection('y_pred_cls', y_pred_cls)

    #
    tf.add_to_collection('final_result', y_pred)

    name = ('model_' + str(conf.iterations) + '_' + str(val_loss)) 
    # Save model weights to disk
    save_path = saver.save(session, conf.model_path + name)
    tf.train.export_meta_graph(conf.model_path + name)
    #write_predictions(test_images, test_ids)
    print("Model saved in file: %s" % save_path)
    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=conf.iterations)

prSt.print_validation_accuracy(data, session, x, y_true, y_pred_cls, show_example_errors=False, show_confusion_matrix=True)

session.close()