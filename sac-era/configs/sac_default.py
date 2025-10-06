import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'sac'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (512, 512)

    config.init_discount = 0.99
    config.final_discount = 0.99
    config.max_steps = 500000

    config.tau = 0.005

    config.init_temperature = 1.0

    return config
