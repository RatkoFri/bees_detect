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


@tf.keras.utils.register_keras_serializable("gradient-accumulator")
class GradientAccumulateOptimizer(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation."""

    def __init__(
        self,
        optimizer="SGD",
        accum_steps=1,
        reduction: str = "MEAN",
        name: str = "GradientAccumulateOptimizer",
        **kwargs
    ):
        """Construct a new GradientAccumulateOptimizer optimizer.

        Adding support for sparse tensors was tricky, but this resource was
        helpful. Note that you need to implement both _resource_apply_sparse()
        and _resource_apply_sparse_duplicate_indices() for it to work as
        intended.

        See here for more information regarding implementation:
        * https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/average_wrapper.py#L93  # noqa

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            reduction: str. Which gradient reduction method to use. Defaults
                to 'SUM'.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulateOptimizer".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        self._optimizer = tf.keras.optimizers.get(optimizer)
        self._accum_steps = accum_steps
        self._reduction = reduction
        self._step = None
        super().__init__(name, **kwargs)

    def _create_slots(self, var_list):
        """Creates slots for optimizer gradients.

        Args:
            List of trainable variables.
        """
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def step(self):
        """The number of training steps this Optimizer has run.
        Initializes step variable if None.

        Returns:
            Current number of optimizer steps.
        """
        if self._step is None:
            with self._distribution_strategy_scope():
                self._step = self.add_weight(
                    "iter",
                    shape=[],
                    initializer="ones",
                    dtype=tf.int64,
                    trainable=False,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                )
            self._weights.append(self._step)
        return self._step

    @step.setter
    def step(self, variable):  # pragma: no cover
        """Sets the step value."""
        if self._step is not None:
            raise RuntimeError(
                "Cannot set `step` to a new Variable after "
                "the Optimizer weights have been created"
            )
        self._step = variable
        self._weights.append(self._step)

    @property
    def gradients(self):  # pragma: no cover
        """The accumulated gradients on the current replica.

        Returns:
            Current gradients in optimizer.
        """
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the"
                "gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Updates weights using gradients.

        Args:
            grads_and_vars: dict containing variables and corresponding
                gradients.
            name: name to set when applying gradients.
            **kwargs: keyword arguments.
        Return:
            Updated weights.
        """
        train_op = super().apply_gradients(grads_and_vars, name, **kwargs)
        with tf.control_dependencies([train_op]):
            with tf.control_dependencies(
                [
                    self._optimizer.iterations.assign_add(
                        tf.cast(
                            tf.where(self.step % self._accum_steps == 0, 1, 0),
                            tf.int64,
                        ),
                        read_value=False,
                    )
                ]
            ):
                return self.step.assign_add(1, read_value=False)

    def _resource_apply_dense(
        self, grad, var, apply_state=None
    ):  # pragma: no cover
        """Performs gradient update on dense tensor.

        Args:
            grad: current gradient.
            var: current variable.
            apply_state: whether to apply X.
        Returns:
            apply_op.
        """
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            accum_gradient.assign_add(
                grad / self._accum_steps,
                use_locking=self._use_locking,
                read_value=False,
            )

        def _apply(accum_gradient, var, apply_state):
            grad = tf.where(
                self.step % self._accum_steps == 0,
                accum_gradient,
                tf.zeros_like(var),
            )

            if "apply_state" in self._optimizer._dense_apply_args:
                train_op = self._optimizer._resource_apply_dense(
                    grad, var, apply_state=apply_state
                )
            else:
                train_op = self.optimizer._resource_apply_dense(grad, var)

            reset_val = tf.where(
                grad == accum_gradient,
                tf.zeros_like(accum_gradient),
                accum_gradient,
            )
            reset_op = accum_gradient.assign(
                reset_val,
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    def _resource_apply_sparse(
        self, grad, var, indices, apply_state=None
    ):  # pragma: no cover
        """Performs gradient update on sparse tensor.

        Args:
            grad: current gradient.
            var: current variable.
            indices: relevant indices to be used for masking the sparse tensor
                during update.
        Returns:
            apply_op.
        """

        accum_gradient = self.get_slot(var, "ga")

        if accum_gradient is not None and grad is not None:
            grad /= tf.cast(self._accum_steps, dtype=grad.dtype)
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply(accum_gradient, var, apply_state):
            grad = tf.where(
                self.step % self._accum_steps == 0,
                accum_gradient,
                tf.zeros_like(var),
            )
            if "apply_state" in self.optimizer._sparse_apply_args:
                train_op = self.optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state,
                )
            else:
                train_op = self.optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices), var, indices
                )

            reset_val = tf.where(
                grad == accum_gradient,
                tf.zeros_like(accum_gradient),
                accum_gradient,
            )
            reset_op = accum_gradient.assign(
                reset_val,
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    # TODO: needs to be updated and tested
    def _resource_apply_sparse_duplicate_indices(
        self, grad, var, indices, apply_state=None
    ):  # pragma: no cover
        """Performs gradient update on sparse tensor.

        Args:
            grad: current gradient.
            var: current variable.
            indices: relevant indices to be used for masking the sparse tensor
                during update.
        Returns:
            apply_op.
        """

        accum_gradient = self.get_slot(var, "ga")

        if accum_gradient is not None and grad is not None:
            grad /= tf.cast(self._accum_steps, dtype=grad.dtype)
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply(accum_gradient, var, apply_state):
            grad = tf.where(
                self.step % self._accum_steps == 0,
                accum_gradient,
                tf.zeros_like(var),
            )
            if "apply_state" in self.optimizer._sparse_apply_args:
                train_op = (
                    self.optimizer._resource_apply_sparse_duplicate_indices(
                        accum_gradient.sparse_read(indices),
                        var,
                        indices,
                        apply_state=apply_state,
                    )
                )
            else:
                train_op = (
                    self.optimizer._resource_apply_sparse_duplicate_indices(
                        accum_gradient.sparse_read(indices), var, indices
                    )
                )

            reset_val = tf.where(
                grad == accum_gradient,
                tf.zeros_like(accum_gradient),
                accum_gradient,
            )
            reset_op = accum_gradient.assign(
                reset_val,
                use_locking=self._use_locking,
                read_value=False,
            )

            return tf.group(train_op, reset_op)

        return _apply(accum_gradient, var, apply_state)

    def reset(self):  # pragma: no cover
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    )
                )

        return tf.group(assign_ops)

    @property
    def optimizer(self):
        """The optimizer that this AccumOptimizer is wrapping."""
        return self._optimizer

    @property
    def iterations(self):
        """Returns current iteration value of optimizer.

        Returns:
            iterations of optimizer."""
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        """Sets the iterations value of optimizer."""
        self._optimizer.iterations = variable

    @property
    def learning_rate(self):  # pragma: no cover
        """Returns the learning rate of the optimizer.

        Returns:
            learning rate of optimizer.
        """
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):  # pragma: no cover
        """Sets the learning rate of the optimizer.

        Args:
            learning_rate: which learning rate to set in the optimizer.
        """
        self._optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        """Returns the configuration as dict."""
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            "accum_steps": self._accum_steps,
            "reduction": self._reduction,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Gets config of original optimizer and deserializes it."""
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)