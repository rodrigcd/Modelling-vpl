import jax


def sigmoid_output_forward_path(W_list, x, n_layers):
    h_list = []
    z_list = []
    for i in range(n_layers):
        z = W_list[i] @ x
        if i == n_layers - 1:
            x = jax.nn.sigmoid(z)
        else:
            x = jax.nn.relu(z)
        h_list.append(x)
        z_list.append(z)
    # Post activation and pre activation values
    return h_list, z_list


def forward_path(W_list, x, n_layers):
    h_list = []
    z_list = []
    for i in range(n_layers):
        z = W_list[i] @ x
        if i == n_layers - 1:
            x = W_list[i] @ x
        else:
            x = jax.nn.relu(W_list[i] @ x)
        h_list.append(x)
        z_list.append(z)
    # Post activation and pre activation values
    return h_list, z_list