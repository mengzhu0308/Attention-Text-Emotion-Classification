#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/27 15:21
@File:          Attention.py
'''

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.initializers import Ones, glorot_uniform

def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = K.cumsum(K.ones(shape=shape, dtype='int32'), axis=-2)
    col_index = K.cumsum(K.ones(shape=shape, dtype='int32'), axis=-1)
    return K.greater_equal(row_index, col_index)

def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tf.logical_and(x, y)

class BaseDenseAttention(Layer):
    def __init__(self, causal=False, dropout=0.0, return_attention_scores=False, **kwargs):
        super(BaseDenseAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.causal = causal
        self.dropout = dropout
        self.return_attention_scores = return_attention_scores

    def _calculate_scores(self, query, key):
        raise NotImplementedError

    def _apply_scores(self, scores, value, scores_mask=None, training=None):
        if scores_mask is not None:
            padding_mask = tf.logical_not(scores_mask)
            scores -= 1e9 * K.cast(padding_mask, K.dtype(scores))
        if training is None:
            training = K.learning_phase()
        weights = K.softmax(scores)

        def dropped_weights():
            return K.dropout(weights, self.dropout)
        weights = K.in_train_phase(dropped_weights, weights, training=training)
        return tf.einsum('bqv, bvd->bqd', weights, value), weights

    def call(self, inputs, mask=None, training=None, **kwargs):
        self._validate_call_args(inputs, mask)
        q, v = inputs[:2]
        k = inputs[2] if len(inputs) > 2 else v
        if mask is not None:
            q_mask, v_mask = mask
        else:
            q_mask = v_mask = None
        scores = self._calculate_scores(q, k)
        if v_mask is not None:
            v_mask = K.expand_dims(v_mask, axis=-2)
        if self.causal:
            scores_shape = K.shape(scores)
            causal_mask_shape = K.concatenate([K.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_scores = self._apply_scores(scores, v, scores_mask=scores_mask, training=training)
        if q_mask is not None:
            q_mask = K.expand_dims(q_mask)
            result *= K.cast(q_mask, K.dtype(result))
        if self.return_attention_scores:
            return [result, attention_scores]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention_scores:
            b, Tq, dim = input_shape[0]
            Tv = input_shape[1][1]
            return [input_shape[0], (b, Tq, Tv)]

        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        if mask is not None:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return q_mask
        return None

    def _validate_call_args(self, inputs, mask):
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError(
                '{} layer must be called on a list of inputs, namely [query, value] '
                'or [query, value, key].'.format(class_name))
        if len(inputs) < 2 or len(inputs) > 3:
            raise ValueError(
                '{} layer accepts inputs list of length 2 or 3, '
                'namely [query, value] or [query, value, key]. '
                'Given length: {}'.format(class_name, len(inputs)))
        if mask is not None:
            if not isinstance(mask, list):
                raise ValueError(
                    '{} layer mask must be a list, '
                    'namely [query_mask, value_mask].'.format(class_name))
            if len(mask) < 2 or len(mask) > len(inputs):
                raise ValueError(
                    '{} layer mask must be a list of length 2, namely [query_mask, '
                    'value_mask]. Given length: {}'.format(class_name, len(mask)))

    def get_config(self):
        config = {
            'causal': self.causal,
            'dropout': self.dropout,
            'return_attention_scores': self.return_attention_scores
        }
        base_config = super(BaseDenseAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Attention(BaseDenseAttention):
    def __init__(self, use_scale=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        if self.use_scale:
            self.scale = self.add_weight(
                name='scale',
                shape=(),
                initializer=Ones())
        else:
            self.scale = None

    def _calculate_scores(self, query, key):
        scores = tf.einsum('bqd, bvd->bqv', query, key)
        if self.scale is not None:
            scores *= self.scale
        return scores

    def get_config(self):
        config = {'use_scale': self.use_scale}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdditiveAttention(BaseDenseAttention):
    def __init__(self, use_scale=True, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        super(AdditiveAttention, self).build(input_shape)
        if self.use_scale:
            self.scale = self.add_weight(
                name='scale',
                shape=(input_shape[1][-1], ),
                initializer=glorot_uniform())
        else:
            self.scale = None

    def _calculate_scores(self, query, key):
        # Reshape tensors to enable broadcasting.
        # Reshape into [batch_size, Tq, 1, dim].
        q_reshaped = K.expand_dims(query, axis=-2)
        # Reshape into [batch_size, 1, Tv, dim].
        k_reshaped = K.expand_dims(key, axis=-3)
        if self.use_scale:
            scale = self.scale
        else:
            scale = 1.
        return K.sum(scale * K.tanh(q_reshaped + k_reshaped), axis=-1)

    def get_config(self):
        config = {'use_scale': self.use_scale}
        base_config = super(AdditiveAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
