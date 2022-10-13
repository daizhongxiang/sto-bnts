from neural_tangents import stax
from neural_tangents.stax import Dense
from jax import jit

from neural_tangents.stax import Conv

def homoscedastic_model(
    W_std,
    b_std,
    width,
    depth,
    activation,
    parameterization
):
    """Construct fully connected NN model and infinite width NTK & NNGP kernel
       function.

    Args:
        W_std (float): Weight standard deviation.
        b_std (float): Bias standard deviation.
        width (int): Hidden layer width.
        depth (int): Number of hidden layers.
        activation (string): Activation function string, 'erf' or 'relu'.
        parameterization (string): Parameterization string, 'ntk' or 'standard'.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`
    """
    act = activation_fn(activation)

    layers_list = [Dense(width, W_std, b_std, parameterization=parameterization)]

    def layer_block():
        return stax.serial(act(), Dense(width, W_std, b_std, parameterization=parameterization))

    for _ in range(depth - 1):
        layers_list += [layer_block()]

    layers_list += [act(), Dense(1, W_std, b_std, parameterization=parameterization)]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers_list)

    apply_fn = jit(apply_fn)

    return init_fn, apply_fn, kernel_fn


def homoscedastic_model_cnn(
    W_std,
    b_std,
    width,
    depth,
    activation,
    parameterization
):
    """Construct fully connected NN model and infinite width NTK & NNGP kernel
       function.

    Args:
        W_std (float): Weight standard deviation.
        b_std (float): Bias standard deviation.
        width (int): Hidden layer width.
        depth (int): Number of hidden layers.
        activation (string): Activation function string, 'erf' or 'relu'.
        parameterization (string): Parameterization string, 'ntk' or 'standard'.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`
    """
    act = activation_fn(activation)

    layers_list = [stax.serial(Conv(width, (3,3), (1,1), "SAME", W_std, b_std), act(), \
                              stax.AvgPool((3, 3), strides=(2, 2)), stax.Flatten())]

    def layer_block():
        return stax.serial(Dense(width, W_std, b_std, parameterization=parameterization))

    for _ in range(depth - 1):
        layers_list += [layer_block()]

    layers_list += [act(), Dense(1, W_std, b_std, parameterization=parameterization)]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers_list)

    apply_fn = jit(apply_fn)

    return init_fn, apply_fn, kernel_fn




def activation_fn(act):
    if act == 'erf':
        return stax.Erf
    elif act == 'relu':
        return stax.Relu
