class CustomDropout(tf.keras.layers.Layer):

    def __init__(self, rate, name="Dropout"):

        super(CustomDropout, self).__init__(name="Dropout")
        self.dp_rate = rate

    def call(self, input, is_train=True):
        
        if (is_train):
            return tf.nn.dropout(input, rate=self.dp_rate)
        return input
        

class Conv2D_Block(tf.keras.layers.Layer):

    def __init__(self, num_blocks, num_filters, kernel_size):

        super(Conv2D_Block, self).__init__()
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.layers = []
        for _ in range(self.num_blocks):
            self.layers.append(tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, padding="same", data_format="channels_last", activation="relu"))
            self.layers.append(tf.keras.layers.BatchNormalization())

    def call(self, inputs):

        output = inputs
        for layer in self.layers:            
            output = layer(output)

        return output

class Encoder_Block(tf.keras.layers.Layer):

    def __init__(self, num_blocks, num_filters, kernel_size, dp_rate):

        super(Encoder_Block, self).__init__()
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dp_rate = dp_rate

        self.layers = []
        self.layers.append(tf.keras.layers.MaxPool2D())
        self.layers.append(CustomDropout(self.dp_rate))
        self.layers.append(Conv2D_Block(self.num_blocks, self.num_filters, self.kernel_size))

    def call(self, input, is_train):

        output = input
        for _, layer in enumerate(self.layers):
            if (layer.name == "Dropout") :
                output = layer(output, is_train)
            else : 
                output = layer(output)

        return output

class UNET(tf.keras.Model):

    def __init__(self, depth, num_blocks, kernel_size, num_filters, dp_rate):
        
        super(UNET, self).__init__()
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_depth = depth

        self.Encoder_layers = []                                                 
        for i in range(self.num_depth):
            self.Encoder_layers.append(Conv2D_Block(self.num_blocks, self.num_filters, self.kernel_size))
            self.Encoder_layers.append(tf.keras.layers.MaxPool2D())
            self.Encoder_layers.append(CustomDropout(dp_rate))
            self.num_filters *= 2
            
        self.Encoder_layers.append(Conv2D_Block(self.num_blocks, self.num_filters, self.kernel_size))

        self.Decoder_layers = []
        for i in range(self.num_depth):
            self.num_filters = self.num_filters/2
            self.Decoder_layers.append(tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, strides=(2,2), padding = "same", data_format="channels_last", 
                                                                       activation="relu"))
            self.Decoder_layers.append(CustomDropout(dp_rate))
            self.Decoder_layers.append(Conv2D_Block(self.num_blocks, self.num_filters, self.kernel_size))
                                                              
        self.Decoder_layers.append(tf.keras.layers.Conv2D(1, 1, padding= "same", data_format="channels_last", activation="sigmoid"))                                                                     
            
    # @tf.function(input_signature=(tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32, name='inp'), tf.TensorSpec(shape=None, dtype=tf.bool, name='is_train'))
    # @tf.function
    def call(self, input, is_train=True):
        
        # input, is_train = inputs

        # print("Input size : ", input.shape)
        encoder_outputs = [input]
        for i, layer in enumerate(self.Encoder_layers):
            if (layer.name == "Dropout"):                
                encoder_outputs.append(layer(encoder_outputs[-1], is_train))
            else :
                encoder_outputs.append(layer(encoder_outputs[-1]))
            # print("Encoder_layer ", i, " output size : ", encoder_outputs[-1].shape)

        # print("\n\n")
        decoder_outputs = [encoder_outputs[-1]]
        # print("Decoder input size : ", decoder_outputs[-1].shape)
        for i in range(self.num_depth):
            decoder_outputs.append(self.Decoder_layers[3*i](decoder_outputs[-1]))
            # print("Decoder layer ", 3*i, " up sampling output size : ", decoder_outputs[-1].shape)
            # print("\t -- Concating input types {}, {}".format(decoder_outputs[-1].shape, encoder_outputs[3*(self.num_depth - i - 1) + 1].shape))
            decoder_outputs[-1] = tf.concat([decoder_outputs[-1], encoder_outputs[3*(self.num_depth - i - 1) + 1]], axis= 3)
            # print("\t -- Concated output ", 3*i, " size : ", decoder_outputs[-1].shape)
            decoder_outputs.append(self.Decoder_layers[3*i+1](decoder_outputs[-1], is_train))
            # print("Decoder layer ", 3*i + 1, "dropout output size : ", decoder_outputs[-1].shape)
            decoder_outputs.append(self.Decoder_layers[3*i+2](decoder_outputs[-1]))
            # print("Decoder input ", 3*i + 2, "conv2d block output size : ", decoder_outputs[-1].shape)

        decoder_outputs.append(self.Decoder_layers[-1](decoder_outputs[-1]))
        # print("Decoder output size : ", decoder_outputs[-1].shape)

        return decoder_outputs[-1]

class UNET_plusplus(tf.keras.Model):

    def __init__(self, num_depth, num_blocks, kernel_size, init_filter_size, dp_rate):

        super(UNET_plusplus, self).__init__()
        self.num_depth = num_depth
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.init_filter_size = init_filter_size
        self.dp_rate = dp_rate
        self.num_filters = self.init_filter_size
                                       
        self.Encoder_layers = []
        self.Encoder_layers.append(Conv2D_Block(self.num_blocks, self.num_filters, self.kernel_size))

        for _ in range(self.num_depth):
            self.num_filters*=2
            self.Encoder_layers.append(Encoder_Block(self.num_blocks, self.num_filters, self.kernel_size, self.dp_rate))

        self.Decoder_layers = []
        for i in range(self.num_depth + 1):
            self.num_filters = self.init_filter_size
            
            Decoder_layer = []
            for j in range(i):
                self.num_filters/=2
                Decoder_layer.append(tf.keras.layers.Conv2DTranspose(self.num_filters, self.kernel_size, strides=(2,2), padding ='same',
                                                                     data_format='channels_last', activation='relu'))
                Decoder_layer.append(CustomDropout(self.dp_rate))
                Decoder_layer.append(Conv2D_Block(self.num_blocks, self.num_filters, self.kernel_size))
            self.Decoder_layers.append(Decoder_layer)

            self.init_filter_size*=2

        # self.output_layers = []
        # for _ in range(self.num_depth):
        #     self.output_layers.append(tf.keras.layers.Conv2D(1, 1, padding='same', data_format='channels_last', activation='sigmoid'))

        # self.final_layer = tf.keras.layers.Average()
        self.final_layer = tf.keras.layers.Conv2D(1, 1, padding='same', data_format='channels_last', activation='sigmoid')

    def call(self, input, is_train):
        
        layer_outputs = [[self.Encoder_layers[0](input)]]
        # print("Layer Output 0, 0 size : ", layer_outputs[0][0].shape)

        for depth in range(1, self.num_depth+1):
            layer_output = []

            # print('Layer {} Outputs '.format(str(depth)) + "  " +"-"*10 + "\n")
            #Encoding
            layer_output.append(self.Encoder_layers[depth](layer_outputs[depth-1][0], is_train))
            # print("Layer Output " + str(depth) + ", 0 size : ", layer_output[-1].shape)
            # print("\n")

            #sub_decoding
            for j in range(depth):
                # print("\t Decoder Output sizes ")

                output = self.Decoder_layers[depth][3*j](layer_output[-1])
                # print("\t Upsample output size : ", output.shape)

                concat_list = [output]
                for k in range(j+1):
                    concat_list.append(layer_outputs[depth-k-1][j-k])
                    # print("\t Concating size ", layer_outputs[depth-k-1][j-k].shape)

                output = tf.concat(concat_list, axis = 3)
                # print("\t Concated output size : ", output.shape)
                output = self.Decoder_layers[depth][3*j+1](output, is_train)
                # print("\t Dropout output size : ", output.shape)
                output = self.Decoder_layers[depth][3*j+2](output)
                # print("\t Conv2D output shape : ", output.shape)
                layer_output.append(output)
                # print("Layer Output " + str(depth) + "," + str(j+1) + " size : ", layer_output[-1].shape)
                # print("\n")

            layer_outputs.append(layer_output)

        # outputs = []
        # for depth in range(self.num_depth): 
        #     outputs.append(self.output_layers[depth](layer_outputs[depth+1][-1]))
        
        # output = self.final_layer(outputs)
        # print("Final Output size : ", output.shape)

        output = self.final_layer(layer_outputs[-1][-1])
        return output

def show_batch(y_true, y_pred, is_test=False):
    # [batch_size x imgh x imw x 1]

    batch_size = y_pred.shape[0]
    rows = batch_size
    cols = 2
    for i in range(batch_size):
        fig, ax = plt.subplots(1,4, figsize = (8, 2))
        img = tf.squeeze(y_pred[i])

        if is_test:
            ax[0].imshow(y_true[i].numpy())
        else :
            ax[0].imshow(y_true[i].numpy(), cmap = "gray", aspect="auto")

        ax[1].imshow(img.numpy(), cmap = "gray", aspect="auto")
        
        ax[2].imshow(tf.cast(img > 0.3, tf.float32).numpy(), cmap="gray", aspect="auto")

        ax[3].hist(tf.keras.backend.flatten(img).numpy(), density=True)

        plt.show()

def eval(y_true, y_pred):

    # IoU
    true_ones = tf.reduce_sum(y_true)
    intersection = y_true * y_pred 
    common_ones = tf.reduce_sum(intersection)
    union = y_true + y_pred - intersection 
    total_ones = tf.reduce_sum(union)

    # accuracy
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    crct_labels = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))
    # print("correct_labels ", crct_labels)
    total_labels = y_true.shape[0]*y_true.shape[1]*y_pred.shape[2]
    # print("total labels ", total_labels)

    return common_ones / true_ones, common_ones / total_ones, crct_labels, total_labels, crct_labels / total_labels

def weighted_cross_entropy(y_true, y_pred, beta):

    # logits = tf.math.log(tf.math.divide(y_pred, 1-y_pred))
    # loss = tf.nn.weighted_cross_entropy_with_logits(y_true, logits, pos_weight=(beta/1-beta))

    loss = (1 - y_true) * (tf.math.log(1 - y_pred)) * (1 - beta) + (y_true) * tf.math.log(y_pred) * (beta) 

    return -tf.reduce_mean(loss)

def dice_loss(y_true, y_pred):
    
    intersection = y_pred * y_true
    union = y_true + y_pred

    return 1 - (2*tf.reduce_sum(intersection) / tf.reduce_sum(union))

def focal_loss(y_true, y_pred, gamma):

    # logits = tf.math.log(tf.divide(y_pred, 1-y_pred))
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred, 1)

    loss = (1 - y_true) * ((y_pred) ** gamma) * (tf.math.log(1 - y_pred)) + (y_true) * ((1-y_pred) ** gamma) * tf.math.log(y_pred) 

    # print(loss)
    return -tf.reduce_mean(loss)

def train(train_dataset, val_dataset, model, lr, num_epochs, loss='cross entropy', save_ckpt=False, save_ckpt_name='', load_ckpt=False, cpkt="", manager=""):

    cce = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    total = BATCHSIZE * IMG_HEIGHT * IMG_WIDTH

    train_losses = []
    val_losses = []
    val_IoUs = []
    val_accs = []
    val_ones_accs = []

    
    if load_ckpt:
        print(manager.latest_checkpoint)
        ckpt.restore(manager.latest_checkpoint)

    for epoch in range(num_epochs):

        epoch_loss = tf.keras.metrics.Mean()

        start_time = time.time()
        for x, y_true in train_dataset:
            with tf.GradientTape() as tape:

                y_pred = model(x, is_train=True)
                # print(y_pred)
                ones = tf.reduce_sum(y_true).numpy() / total
                # print("fraction of ones : {:.5f}".format(ones))
                # print("True shape {} and type {}, pred shape {}, type {} and max value {}".format(y_true.shape, y_true.dtype, y_pred.shape, y_pred.dtype, tf.reduce_max(y_pred)))
                
                if loss == 'cross entropy':
                    loss_value = cce(y_true, y_pred)
                    # print(ones)
                    # loss_value = weighted_cross_entropy(y_true, y_pred, (1-ones))
                elif loss == 'combined loss':
                    loss_value = dice_loss(y_true, y_pred) + cce(y_true, y_pred)
                elif loss == 'dice loss':
                    loss_value = dice_loss(y_true, y_pred)
                else:
                    loss_value = focal_loss(y_true, y_pred, 2)
                
                # print("Loss is {:.9f}".format(loss_value))
                grads = tape.gradient(loss_value, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss.update_state(loss_value)
            
            # o1, o2, o3, o4, o5 = eval(y_true, y_pred)
            # print("Train : ones_accc : {:.5f}, IoU : {:.5f}, crct : {:.5f}, total : {:.5f}, acc : {:.5f}".format(o1, o2, o3, o4, o5))

            # if (epoch % 10 == 0):
            #     show_batch(tf.squeeze(y_true), tf.squeeze(y_pred))

        
        train_losses.append(epoch_loss.result())
        # print("\n" + "=======================================================================")
        print("Epoch {:03d} Time : {:.3f}, :- Loss - {:.6f}".format(epoch, time.time() - start_time, epoch_loss.result()))

        # Validation        
        Val_loss = tf.keras.metrics.Mean()
        Val_ones_acc = tf.keras.metrics.Mean()
        Val_IoU = tf.keras.metrics.Mean()
        Val_crct = tf.keras.metrics.Mean()
        Val_total = tf.keras.metrics.Mean()
        Val_acc = tf.keras.metrics.Mean()

        start_time = time.time()
        for x, y_true in val_dataset:   

            y_pred = model(x, is_train=False)

            if loss == 'cross entropy':
                val_loss = cce(y_true, y_pred)
                # val_loss = weighted_cross_entropy(y_true, y_pred, (1-ones))
            elif loss == 'dice loss':
                val_loss = dice_loss(y_true, y_pred)
            else:
                val_loss = focal_loss(y_true, y_pred, 2)            
            Val_loss.update_state(val_loss)

            val_ones_acc, val_IoU, val_crct, val_total, val_acc = eval(y_true, y_pred)
            Val_ones_acc.update_state(val_ones_acc)
            Val_IoU.update_state(val_IoU)
            Val_crct.update_state(val_crct)
            Val_total.update_state(val_total)
            Val_acc.update_state(val_acc)
            # show_batch(tf.squeeze(y_true), tf.squeeze(y_pred))

        val_losses.append(Val_loss.result())
        val_ones_accs.append(Val_ones_acc.result())
        val_accs.append(Val_acc.result())
        val_IoUs.append(Val_IoU.result())
        print("\nEval Time : {:.3f}, loss : {:.6f}, ones_accc : {:.5f}, IoU : {:.5f}, crct : {:.5f}, total : {:.5f}, acc : {:.5f}".
              format(time.time() - start_time, Val_loss.result(), Val_ones_acc.result(), Val_IoU.result(), Val_crct.result(), Val_total.result(), Val_acc.result()))
        print("========latest_checkpoint=================================================================" + "\n")
    
    if save_ckpt:
        save_path = manager.save()
        print(save_path)

    return model, train_losses, val_losses, val_ones_accs, val_IoUs, val_accs