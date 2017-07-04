from tfinterface.supervised import SigmoidClassifier
import tensorflow as tf
import tfinterface as ti


class Model(SigmoidClassifier):

    def __init__(self, *args, **kwargs):
        self._activation = kwargs.pop("activation", None)
        self._weight_decay = kwargs.pop("weight_decay", 1E-4)

        self._growth_rate = kwargs.pop("growth_rate", 4)
        self._depth = kwargs.pop("depth", 5)

        super(Model, self).__init__(*args, **kwargs)

    def get_logits(self, inputs):

        net = inputs.features
        self.layers = []

        for i in range(self._depth):
            layer = tf.layers.dense(net, self._growth_rate, activation=self._activation)
            net = tf.concat([net, layer],  axis=1)

            self.layers.append(layer)

        net = tf.layers.dense(net, 1)
        self.layers.append(net)

        return net

    def get_predictions(self, *args, **kwargs):

        predictions = super(Model, self).get_predictions(*args, **kwargs)
        return predictions

    def get_loss(self, *args, **kwargs):

        loss_preds = super(Model, self).get_loss(*args, **kwargs)

        loss_reg = map(tf.nn.l2_loss, self.get_variables())
        loss_reg = sum(loss_reg)

        return loss_preds + self._weight_decay * loss_reg
