import chainer
import chainer.functions as F
import chainer.links as L


class Alex_RNNLM(chainer.Chain):

    """
    """
    insize = 227

    def __init__(self, n_units, train=True):

        super(Alex_RNNLM, self).__init__(
            conv1=L.Convolution2D(3,  12, 11, stride=4),
            conv2=L.Convolution2D(12, 32,  5, pad=2),
            conv3=L.Convolution2D(32, 48,  3, pad=1),
            conv4=L.Convolution2D(48, 48,  3, pad=1),
            conv5=L.Convolution2D(48, 32,  3, pad=1),
            # 256 * 3 * 12 = 9216
            # fc6=L.Linear(9216, 4096),
            # fc7=L.Linear(4096, 4096),
            # fc8=L.Linear(4096, n_units),
            fc6=L.Linear(1152, 45),
            fc7=L.Linear(45, 45),
            fc8=L.Linear(45, n_units),
        )

        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        self.clear()
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        return h
