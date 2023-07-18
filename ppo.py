import os
import os.path

import numpy as np
import tensorflow as tf

from gridworld.gridworld import GridWorld

ENTROPY_LOSS = 0.01
LOSS_CLIPPING = 0.2
L2_REG = 1e-4
N_FILTERS = 32
N_LAYERS = 3
SHUTDOWN = False


class ActorModel:
    def __init__(self, network_def, input_shape, action_space):
        self.action_space = action_space
        inputs, x = network_def(input_shape)
        outputs = tf.keras.layers.Dense(self.action_space, activation='softmax')(x)
        self.Actor = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.Actor.compile(loss=self.ppo_loss, optimizer=tf.optimizers.Adam())

    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:, 1 + self.action_space:]
        prob = actions * y_pred
        old_prob = actions * prediction_picks
        prob = tf.clip_by_value(prob, 1e-10, 1.0)
        old_prob = tf.clip_by_value(old_prob, 1e-10, 1.0)
        ratio = tf.math.exp(tf.math.log(prob) - tf.math.log(old_prob))
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING) * advantages
        actor_loss = -tf.math.reduce_mean(tf.math.minimum(p1, p2))
        entropy = -(y_pred * tf.math.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * tf.math.reduce_mean(entropy)
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.Actor.trainable_variables])
        l2_regularization = L2_REG * l2_loss
        return actor_loss - entropy + l2_regularization

    def fit(self, states, y_true):
        self.Actor.fit(states, y_true, epochs=16, verbose=0)

    def predict(self, state):
        return self.Actor.predict(state)

    def load(self):
        self.Actor = tf.keras.models.load_model('actor.h5', compile=False)
        self.Actor.compile(loss=self.ppo_loss, optimizer=tf.optimizers.Adam())

    def save(self):
        self.Actor.save('actor.h5')


class CriticModel:
    def __init__(self, network_def, input_shape):
        inputs, x = network_def(input_shape)
        outputs = tf.keras.layers.Dense(1, activation=None)(x)
        self.Critic = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.Critic.compile(loss=tf.keras.losses.Huber(), optimizer=tf.optimizers.Adam())

    def fit(self, states, target):
        self.Critic.fit(states, target, epochs=16, verbose=0)

    def predict(self, state):
        return self.Critic.predict(state)

    def load(self):
        self.Critic = tf.keras.models.load_model('critic.h5', compile=False)
        self.Critic.compile(loss=tf.keras.losses.Huber(), optimizer=tf.optimizers.Adam())

    def save(self):
        self.Critic.save('critic.h5')


def grid_network(input_shape):
    inputs = tf.keras.layers.Input(input_shape, name='inputs')
    x = tf.keras.layers.Conv2D(N_FILTERS, 3, padding='same', activation='relu')(inputs)
    for _ in range(N_LAYERS):
        fx = tf.keras.layers.Conv2D(N_FILTERS, 3, padding='same')(x)
        x = tf.keras.layers.Add()([x, fx])
        x = tf.keras.layers.ReLU()(x)
    x = tf.reduce_sum(x * inputs[:, :, :, :1], [1, 2])
    x = tf.keras.layers.Dense(N_FILTERS, activation=None)(x)
    return inputs, x


def get_advantages(rewards, rewards_p, dones, gamma=0.99):
    gaes = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - dones[t]) * rewards_p[t] - rewards_p[t]
        gaes[t] = last_gae = delta + gamma * (1 - dones[t]) * gamma * last_gae

    advantages = gaes.copy()
    advantages -= advantages.mean()
    advantages /= (advantages.std() + 1e-8)

    target = rewards_p + gaes
    target -= target.mean()
    target /= (target.std() + 1e-8)

    return np.vstack(advantages), np.vstack(target)


def replay(actor, critic, states, actions, rewards, actions_ps, dones):
    states = np.vstack(states)
    actions = np.vstack(actions)
    actions_ps = np.vstack(actions_ps)
    rewards_p = critic.predict(states)
    advantages, target = get_advantages(rewards, np.squeeze(rewards_p), dones)
    y_true = np.hstack([advantages, actions_ps, actions])
    actor.fit(states, y_true)
    critic.fit(states, target)


def act(state, actor, action_size):
    action_p = actor.predict(state)[0]
    action = np.random.choice(action_size, p=action_p)
    action_onehot = np.zeros([action_size])
    action_onehot[action] = 1
    return action, action_onehot, action_p


def main():
    env = GridWorld()
    state_size = (16, 16, 4)
    action_size = 5

    actor = ActorModel(network_def=grid_network, input_shape=state_size, action_space=action_size)
    critic = CriticModel(network_def=grid_network, input_shape=state_size)

    if os.path.isfile('actor.h5'):
        actor.load()
        critic.load()

    episode = 0
    running_reward = 0

    while True:
        obs, _ = env.reset(episode, running_reward)
        obs = tf.expand_dims(obs, 0)
        states, actions, rewards, actions_ps, dones = [], [], [], [], []
        while True:
            env.render()
            action, action_onehot, action_p = act(obs, actor, action_size)
            next_obs, reward, done, _, _ = env.step(action)
            states.append(obs)
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            actions_ps.append(action_p)
            obs = tf.expand_dims(next_obs, 0)
            if done:
                break

        replay(actor, critic, states, actions, rewards, actions_ps, dones)
        running_reward = running_reward * 0.99 + sum(rewards) * 0.01
        print(f"running reward at episode {episode} is {running_reward}")
        episode += 1

        if episode > 50 and running_reward > 5:
            break

    actor.save()
    critic.save()

    if SHUTDOWN:
        os.system("shutdown /s /t 1")


if __name__ == '__main__':
    main()
