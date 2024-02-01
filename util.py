from Capsule_MPNN import *

def get_test_validation(train_p, train_d, train_y, dti):
    if dti == 'data/ncDR.csv':
        train_p_train, train_p_val = train_p[:6238], train_p[6238:7130]
        train_d_train, train_d_val = np.array(train_d)[:6238], np.array(
            train_d)[6238:7130]
        train_y_train, train_y_val = train_y[:6238], train_y[6238:7130]

    if dti == 'data/RNAInter.csv':
        train_p_train, train_p_val = train_p[:8034], train_p[8034:9182]
        train_d_train, train_d_val = np.array(train_d)[:8034], np.array(
            train_d)[8034:9182]
        train_y_train, train_y_val = train_y[:8034], train_y[8034:9182]

    if dti == 'data/SM2miR1.csv':
        train_p_train, train_p_val = train_p[:1872], train_p[1872:2140]
        train_d_train, train_d_val = np.array(train_d)[:1872], np.array(
            train_d)[1872:2140]
        train_y_train, train_y_val = train_y[:1872], train_y[1872:2140]
    if dti == 'data/SM2miR2.csv':

        train_p_train, train_p_val = train_p[:2748], train_p[2748:3140]
        train_d_train, train_d_val = np.array(train_d)[:2748], np.array(
            train_d)[2748:3140]
        train_y_train, train_y_val = train_y[:2748], train_y[2748:3140]

    if dti == 'data/SM2miR3.csv':
        train_p_train, train_p_val = train_p[:3166], train_p[3166:3618]
        train_d_train, train_d_val = np.array(train_d)[:3166], np.array(
            train_d)[3166:3618]
        train_y_train, train_y_val = train_y[:3166], train_y[3166:3618]

    return train_p_train,train_p_val,train_d_train,train_d_val,train_y_train,train_y_val



def get_test_validation_MPNN(train_p, train_d, train_y, dti):
    if dti == 'data/ncDR.csv':
        train_p_train, train_p_val = train_p[:6238], train_p[6238:7130]
        train_d0, train_d1, train_d2 = train_d
        train_d0_train, train_d0_val = np.array(train_d0)[:6238], \
            np.array(train_d0)[6238:7130]
        train_d1_train, train_d1_val = np.array(train_d1)[:6238], \
            np.array(train_d1)[6238:7130]
        train_d2_train, train_d2_val = np.array(train_d2)[:6238], \
            np.array(train_d2)[6238:7130]
        train_d_train = (
            tf.ragged.constant(train_d0_train, dtype=tf.float32),
            tf.ragged.constant(train_d1_train, dtype=tf.float32),
            tf.ragged.constant(train_d2_train, dtype=tf.int64))
        train_d_val = (tf.ragged.constant(train_d0_val, dtype=tf.float32),
                       tf.ragged.constant(train_d1_val, dtype=tf.float32),
                       tf.ragged.constant(train_d2_val, dtype=tf.int64))

        train_y_train, train_y_val = train_y[:6238], train_y[6238:7130]
        train_dataset = MPNNDataset(train_p_train, train_d_train,
                                    train_y_train)
        valid_dataset = MPNNDataset(train_p_val, train_d_val, train_y_val)

    if dti == 'data/RNAInter.csv':
        train_p_train, train_p_val = train_p[:8034], train_p[8034:9182]
        train_d0, train_d1, train_d2 = train_d
        train_d0_train, train_d0_val = np.array(train_d0)[:8034], \
            np.array(train_d0)[8034:9182]
        train_d1_train, train_d1_val = np.array(train_d1)[:8034], \
            np.array(train_d1)[8034:9182]
        train_d2_train, train_d2_val = np.array(train_d2)[:8034], \
            np.array(train_d2)[8034:9182]
        train_d_train = (
            tf.ragged.constant(train_d0_train, dtype=tf.float32),
            tf.ragged.constant(train_d1_train, dtype=tf.float32),
            tf.ragged.constant(train_d2_train, dtype=tf.int64))
        train_d_val = (tf.ragged.constant(train_d0_val, dtype=tf.float32),
                       tf.ragged.constant(train_d1_val, dtype=tf.float32),
                       tf.ragged.constant(train_d2_val, dtype=tf.int64))

        train_y_train, train_y_val = train_y[:8034], train_y[8034:9182]
        train_dataset = MPNNDataset(train_p_train, train_d_train,
                                    train_y_train)
        valid_dataset = MPNNDataset(train_p_val, train_d_val, train_y_val)

    if dti == 'data/SM2miR1.csv':
        train_p_train, train_p_val = train_p[:1872], train_p[1872:2140]
        train_d0, train_d1, train_d2 = train_d
        train_d0_train, train_d0_val = np.array(train_d0)[:1872], \
            np.array(train_d0)[1872:2140]
        train_d1_train, train_d1_val = np.array(train_d1)[:1872], \
            np.array(train_d1)[1872:2140]
        train_d2_train, train_d2_val = np.array(train_d2)[:1872], \
            np.array(train_d2)[1872:2140]
        train_d_train = (
            tf.ragged.constant(train_d0_train, dtype=tf.float32),
            tf.ragged.constant(train_d1_train, dtype=tf.float32),
            tf.ragged.constant(train_d2_train, dtype=tf.int64))
        train_d_val = (tf.ragged.constant(train_d0_val, dtype=tf.float32),
                       tf.ragged.constant(train_d1_val, dtype=tf.float32),
                       tf.ragged.constant(train_d2_val, dtype=tf.int64))

        train_y_train, train_y_val = train_y[:1872], train_y[1872:2140]
        train_dataset = MPNNDataset(train_p_train, train_d_train,
                                    train_y_train)
        valid_dataset = MPNNDataset(train_p_val, train_d_val, train_y_val)

    if dti == 'data/SM2miR2.csv':
        train_p_train, train_p_val = train_p[:2748], train_p[2748:3140]
        train_d0, train_d1, train_d2 = train_d
        train_d0_train, train_d0_val = np.array(train_d0)[:2748], \
            np.array(train_d0)[2748:3140]
        train_d1_train, train_d1_val = np.array(train_d1)[:2748], \
            np.array(train_d1)[2748:3140]
        train_d2_train, train_d2_val = np.array(train_d2)[:2748], \
            np.array(train_d2)[2748:3140]
        train_d_train = (
            tf.ragged.constant(train_d0_train, dtype=tf.float32),
            tf.ragged.constant(train_d1_train, dtype=tf.float32),
            tf.ragged.constant(train_d2_train, dtype=tf.int64))
        train_d_val = (tf.ragged.constant(train_d0_val, dtype=tf.float32),
                       tf.ragged.constant(train_d1_val, dtype=tf.float32),
                       tf.ragged.constant(train_d2_val, dtype=tf.int64))

        train_y_train, train_y_val = train_y[:2748], train_y[2748:3140]
        train_dataset = MPNNDataset(train_p_train, train_d_train,
                                    train_y_train)
        valid_dataset = MPNNDataset(train_p_val, train_d_val, train_y_val)

    if dti == 'data/SM2miR3.csv':
        train_p_train, train_p_val = train_p[:3166], train_p[3166:3618]
        train_d0, train_d1, train_d2 = train_d
        train_d0_train, train_d0_val = np.array(train_d0)[:3166], \
            np.array(train_d0)[3166:3618]
        train_d1_train, train_d1_val = np.array(train_d1)[:3166], \
            np.array(train_d1)[3166:3618]
        train_d2_train, train_d2_val = np.array(train_d2)[:3166], \
            np.array(train_d2)[3166:3618]
        train_d_train = (
            tf.ragged.constant(train_d0_train, dtype=tf.float32),
            tf.ragged.constant(train_d1_train, dtype=tf.float32),
            tf.ragged.constant(train_d2_train, dtype=tf.int64))
        train_d_val = (tf.ragged.constant(train_d0_val, dtype=tf.float32),
                       tf.ragged.constant(train_d1_val, dtype=tf.float32),
                       tf.ragged.constant(train_d2_val, dtype=tf.int64))

        train_y_train, train_y_val = train_y[:3166], train_y[3166:3618]
        train_dataset = MPNNDataset(train_p_train, train_d_train,
                                    train_y_train)
        valid_dataset = MPNNDataset(train_p_val, train_d_val, train_y_val)

    return train_y_val,train_dataset,valid_dataset