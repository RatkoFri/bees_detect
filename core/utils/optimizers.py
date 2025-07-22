# -*- coding: utf-8 -*-
#################################################################
# Code from https://github.com/LJNL/accum_optimizer_for_keras
#################################################################
import tensorflow as tf


class Accumulative(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, accum_steps=1, name='accum', **kwargs):
        self.optimizer = optimizer
        self.name = name
        self.learning_rate = optimizer.learning_rate.numpy()
        self._learning_rate = self._build_learning_rate(self.learning_rate)
        super(Accumulative, self).__init__(name, **kwargs)
        
        self.learning_rate = self.optimizer.learning_rate
        self._learning_rate = self._build_learning_rate(self.learning_rate)

        with tf.name_scope(self.__class__.__name__):
            self.accum_steps = accum_steps
            self.iterations = tf.Variable(0, dtype='int64', name='iterations')
            self.cond = tf.equal(self.iterations % self.accum_steps, 0)
            self.learning_rate = self.optimizer.learning_rate

            self.optimizer.learning_rate = tf.cond(self.cond, lambda: self.optimizer.learning_rate.value(), lambda: 0.)
            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, tf.cond(self.cond, lambda: value.value(), lambda: 1 - 1e-7))
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)

            self._create_slots = self.optimizer._create_slots
            self._resource_apply_dense = self.optimizer._resource_apply_dense

            def get_gradients(loss, params):
                return [ag / self.accum_steps for ag in self.accum_grads]

            self.optimizer.get_gradients = get_gradients

    def get_updates(self, loss, params):
        self.iterations = tf.add(self.iterations, 1)
        self.optimizer.iterations = tf.add(self.optimizer.iterations, tf.cast(self.cond, 'int64'))
        self.updates = [
            self.iterations,
            self.optimizer.iterations
        ]
        # gradient accumulation
        self.accum_grads = [tf.zeros(p.shape, dtype=p.dtype) for p in params]
        grads = self.get_gradients(loss, params)

        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(ag=tf.cond(self.cond, lambda: g, lambda: ag + g))

        # inheriting updates of original optimizer
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        return self.updates

    def get_config(self):
        iterations = self.iterations.numpy()
        self.iterations = 0
        config = self.optimizer.get_config()
        self.iterations = iterations
        return config

class AccumulativeV2(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, accum_steps=1, name='accum', **kwargs):
        self.optimizer = optimizer
        self.name = name
        super(Accumulative, self).__init__(name, **kwargs)
        self.learning_rate = self.optimizer.learning_rate
        self._learning_rate = self._build_learning_rate(self.learning_rate)

        with tf.name_scope(self.__class__.__name__):
            self.accum_steps = accum_steps
            self.iterations = tf.Variable(0, dtype='int64', name='iterations')
            self.cond = tf.equal(self.iterations % self.accum_steps, 0)
            self.learning_rate = self.optimizer.learning_rate

            self.optimizer.learning_rate = tf.cond(self.cond, lambda: self.optimizer.learning_rate.value(), lambda: 0.)
            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, tf.cond(self.cond, lambda: value.value(), lambda: 1 - 1e-7))
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)

    
            def get_gradients(loss, params):
                return [ag / self.accum_steps for ag in self.accum_grads]

            self.optimizer.get_gradients = get_gradients

    def get_updates(self, loss, params):
        self.iterations = tf.add(self.iterations, 1)
        self.optimizer.iterations = tf.add(self.optimizer.iterations, tf.cast(self.cond, 'int64'))
        self.updates = [
            self.iterations,
            self.optimizer.iterations
        ]
        # gradient accumulation
        self.accum_grads = [tf.zeros(p.shape, dtype=p.dtype) for p in params]
        grads = self.get_gradients(loss, params)

        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(ag=tf.cond(self.cond, lambda: g, lambda: ag + g))

        # inheriting updates of original optimizer
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        return self.updates

    def get_config(self):
        iterations = self.iterations.numpy()
        self.iterations = 0
        config = self.optimizer.get_config()
        self.iterations = iterations
        return config
    
import tensorflow as tf
from typing import Iterable, List, Tuple


class GAOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        gradient_accumulation_steps: int = 1,
        name: str = 'GAOptimizer',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._learning_rate = self._build_learning_rate(optimizer.learning_rate)


    def apply_gradients(
        self,
        grads_and_vars: Iterable[Tuple[tf.Tensor, tf.Variable]],
        *args,
        **kwargs
    ):
        grads_and_vars = list(grads_and_vars)
        vars = [var for _, var in grads_and_vars]
        if not hasattr(self, '_built') or not self._built:
            self.build(vars)

        self.step.assign_add(1)
        should_apply = tf.equal(self.step % self.gradient_accumulation_steps, 0)

        # update accumulated gradients
        self._update_accumulated_grads(grads_and_vars)

        # apply gradients
        def _cross_replica_apply_gradients(strategy, grads_and_vars):
            def _apply_fn():
                strategy.extended.call_for_each_replica(
                    self._apply_accumulated_grads,
                    args=(grads_and_vars, *args), kwargs=kwargs)
            tf.cond(should_apply, _apply_fn, lambda: None)

        tf.distribute.get_replica_context().merge_call(
            _cross_replica_apply_gradients, args=(grads_and_vars,))

        # reset accumulated gradients if necessary
        tf.cond(should_apply, self._reset_accumulated_grads, lambda: None)

        return self.optimizer.iterations

    def _update_accumulated_grads(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]]
    ):
        for i, (grad, _) in enumerate(grads_and_vars):
            self.accumulated_grads[i].assign_add(grad)

    def _apply_accumulated_grads(
        self,
        grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
        *args,
        **kwargs
    ):
        accumulated_grads_and_vars = [
            (
                self.accumulated_grads[i] / tf.cast(
                    self.gradient_accumulation_steps,
                    self.accumulated_grads[i].dtype),
                var
            )
            for i, (_, var) in enumerate(grads_and_vars)
        ]
        self.optimizer.apply_gradients(
            accumulated_grads_and_vars, *args, **kwargs)

    def _reset_accumulated_grads(self):
        for grad in self.accumulated_grads:
            grad.assign(tf.zeros_like(grad))

    def build(self, var_list: List[tf.Variable]):
        super().build(var_list)
        self.optimizer.build(var_list)
        self.accumulated_grads = [
            tf.Variable(
                initial_value=tf.zeros_like(var),
                trainable=False,
                aggregation=tf.VariableAggregation.NONE)
            for var in var_list
        ]
        self.step = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int64,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self._built = True