
import tensorflow as tf

def dice_coef(y_true, y_pred):
    import keras.backend as K
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
	
#######################################################################################
	
def fbeta_coef(y_true, y_pred, alpha, beta, weights=False, smooth=1):    
    
    y_true_f = K.flatten(y_true)
    y_true_f_r = K.flatten(1. - y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f_r = K.flatten(1. - y_pred)
    
    if len(weights) < 1:
        weights = 1.
        
    else:
        weights = K.variable([11.])
    
    intersection = K.sum(y_pred_f * y_true_f *  weights)
    
    fp = K.sum(y_pred_f * y_true_f_r)
    fn = K.sum(y_pred_f_r * y_true_f *  weights)

    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)
	
def fbeta_coef_loss(alpha, beta, weights=False):
    def fbeta(y_true, y_pred):
        return -fbeta_coef(y_true, y_pred, alpha, beta, weights)
    return fbeta

fbeta_loss = fbeta_coef_loss(alpha=0.3, beta=0.7, weights=[])

#######################################################################################

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, name=None, scope=None):

    logits = tf.convert_to_tensor(y_pred)
    onehot_labels = tf.convert_to_tensor(y_true)
    precise_logits = tf.cast(logits, tf.float32) if (
                    logits.dtype == tf.float16) else logits
    onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
    #predictions = tf.nn.sigmoid(precise_logits)
    predictions = precise_logits
    predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
    # add small value to avoid 0
    epsilon = 1e-8
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
    losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                                 name=name, axis=1)
    return losses
	
#######################################################################################
	
def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalized_dice_loss(ground_truth,
                          prediction,                          
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    
#     print np.shape(prediction)
#     print np.shape(ground_truth)
#     print np.shape(ground_truth[..., -1])
    
#     print tf.shape(prediction)
#     print tf.shape(prediction)[-1]
    
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
#     one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    one_hot = labels_to_one_hot(ground_truth)
    
#    print(np.shape(one_hot))

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
#         ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
#         intersect = tf.sparse_reduce_sum(one_hot * prediction,
#                                          reduction_axes=[0])
        ref_vol = tf.reduce_sum(one_hot, 0)
        intersect = tf.reduce_sum(one_hot * prediction, 0)
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
        tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    return 1 - generalised_dice_score