import os
import tensorflow as tf
from keras.callbacks import TensorBoard
import shutil
from sys import platform

class TrainValTensorBoard(TensorBoard):
    if platform == "linux" or platform == "linux2":
        # linux
        logDir = './/logs'
    elif platform == "darwin":
        # OS X
        logDir = './/logs'
    elif platform == "win32":
        # Windows...
        logDir = '.\\logs'
    
    def __init__(self, log_dir=logDir, **kwargs):
		# Remove all previous log files
        shutil.rmtree(log_dir)
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics

        #Alex: Edits to above. rename so that same name as training. Now it is working!
        repls = ('val_loss', 'epoch_loss'), ('val_acc', 'epoch_acc')
        val_logs = {reduce(lambda a, kv: a.replace(*kv), repls, k): v for k, v in logs.items() if k.startswith('val_')}

        #val_logs = {k.replace('val_loss', 'epoch_loss'): v for k, v in logs.items() if k.startswith('val_')}

        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()


        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)


    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()