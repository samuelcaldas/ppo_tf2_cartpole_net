using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//import tensorflow as tf
//from tensorflow import keras
//import tensorflow_probability as tfp
//import numpy as np
//import gym
//import datetime as dt

namespace ppo_tf2_cartpole
{
    class Model
    { //(keras.Model)
        static void Model(var num_actions)
        {
            super().__init__();
            this.num_actions = num_actions;
            this.dense1 = keras.layers.Dense(64, activation: "relu",
                                             kernel_initializer: keras.initializers.he_normal());
            this.dense2 = keras.layers.Dense(64, activation: "relu",
                                             kernel_initializer: keras.initializers.he_normal());
            this.value = keras.layers.Dense(1);
            this.policy_logits = keras.layers.Dense(num_actions);

        }
        static void call(var inputs)
        {
            var x = this.dense1(inputs);
            x = this.dense2(x);
            return this.value(x), this.policy_logits(x);
        }
        static void action_value(var state)
        {
            var (value, logits) = this.predict_on_batch(state);
            var dist = tfp.distributions.Categorical(logits: logits);
            var action = dist.sample();
            return (action, value);
        }
    }
    class Program
    {
        static var STORE_PATH = "C:\\Users\\andre\\TensorBoard\\PPOCartpole";
        static var CRITIC_LOSS_WEIGHT = 0.5;
        static var ENTROPY_LOSS_WEIGHT = 0.01;
        static var ENT_DISCOUNT_RATE = 0.995;
        static var BATCH_SIZE = 64;
        static var GAMMA = 0.99;
        static var CLIP_VALUE = 0.2;
        static var LR = 0.001;

        static var NUM_TRAIN_EPOCHS = 10;

        static var env = gym.make("CartPole-v0");
        static var state_size = 4;
        static var num_actions = env.action_space.n;

        static var ent_discount_val = ENTROPY_LOSS_WEIGHT;

        static void critic_loss(var discounted_rewards, var value_est)
        {
            return tf.cast(tf.reduce_mean(keras.losses.mean_squared_error(discounted_rewards, value_est)) * CRITIC_LOSS_WEIGHT,
                           tf.float32);
        }
        static void entropy_loss(var policy_logits, var ent_discount_val)
        {
            var probs = tf.nn.softmax(policy_logits);
            var entropy_loss = -tf.reduce_mean(keras.losses.categorical_crossentropy(probs, probs));
            return entropy_loss * ent_discount_val;
        }
        static void actor_loss(var advantages, var old_probs, var action_inds, var policy_logits)
        {
            var probs = tf.nn.softmax(policy_logits);
            var new_probs = tf.gather_nd(probs, action_inds);

            var ratio = new_probs / old_probs;

            var policy_loss = -tf.reduce_mean(tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - CLIP_VALUE, 1.0 + CLIP_VALUE) * advantages
        ));
            return policy_loss;
        }
        static void train_model(var action_inds, var old_probs, var states, var advantages, var discounted_rewards, var optimizer, var ent_discount_val)
        {
            using (var tape = tf.GradientTape())
            {
                var (values, policy_logits) = model.call(tf.stack(states));
                var act_loss = actor_loss(advantages, old_probs, action_inds, policy_logits);
                var ent_loss = entropy_loss(policy_logits, ent_discount_val);
                var c_loss = critic_loss(discounted_rewards, values);
                var tot_loss = act_loss + ent_loss + c_loss;
            }
            var grads = tape.gradient(tot_loss, model.trainable_variables);
            optimizer.apply_gradients(zip(grads, model.trainable_variables));
            return (tot_loss, c_loss, act_loss, ent_loss);
        }
        static void get_advantages(var rewards, var dones, var values, var next_value)
        {
            var discounted_rewards = np.array(rewards + [next_value[0]]);

            for (var t in reversed(range(len(rewards))))
            {
                discounted_rewards[t] = rewards[t] + GAMMA * discounted_rewards[t + 1] * (1 - dones[t]);
            }
            discounted_rewards = discounted_rewards[":-1"];
            // advantages are bootstrapped discounted rewards - values, using Bellman's equation
            var advantages = discounted_rewards - np.stack(values)[":, 0"];
            // standardise advantages
            advantages -= np.mean(advantages);
            advantages /= (np.std(advantages) + 1e-10);
            // standardise rewards too
            discounted_rewards -= np.mean(discounted_rewards);
            discounted_rewards /= (np.std(discounted_rewards) + 1e-8);
            return (discounted_rewards, advantages);
        }
        static void Main(string[] args)
        {
            var model = new Model(num_actions);
            var optimizer = keras.optimizers.Adam(learning_rate: LR);

            var train_writer = tf.summary.create_file_writer(STORE_PATH + f"/PPO-CartPole_{dt.datetime.now().strftime(" % d % m % Y % H % M")}");

            var num_steps = 10000000;
            var episode_reward_sum = 0;
            var state = env.reset();
            var episode = 1;
            var total_loss = Null;
            for (var step in range(num_steps))
            {
                var rewards = new List<double>();
                var actions = new List<double>();
                var values = new List<double>();
                var states = new List<double>();
                var dones = new List<double>();
                var probs = new List<double>();
                for (var _ in range(BATCH_SIZE))
                {
                    var (_, policy_logits) = model(state.reshape(1, -1));

                    var (action, value) = model.action_value(state.reshape(1, -1));
                    var (new_state, reward, done, _) = env.step(action.numpy()[0]);

                    actions.Add(action);
                    values.Add(value[0]);
                    states.Add(state);
                    dones.Add(done);
                    probs.Add(policy_logits);
                    episode_reward_sum += reward;

                    state = new_state;

                    if (done)
                    {
                        rewards.Add(0.0);
                        state = env.reset();
                        if (total_loss is not Null)
                        {
                            Console.WriteLine(f"Episode: {episode}, latest episode reward: {episode_reward_sum}, "
                                  f"total loss: {np.mean(total_loss)}, critic loss: {np.mean(c_loss)}, "
                                  f"actor loss: {np.mean(act_loss)}, entropy loss {np.mean(ent_loss)}");
                        }
                        using (train_writer.as_default())
                        {
                            tf.summary.scalar("rewards", episode_reward_sum, episode);
                        }
                        episode_reward_sum = 0;

                        episode += 1;
                    }
                    else
                    {
                        rewards.Add(reward);
                    }
                }
                var (_, next_value) = model.action_value(state.reshape(1, -1));
                var (discounted_rewards, advantages) = get_advantages(rewards, dones, values, next_value[0]);

                actions = tf.squeeze(tf.stack(actions));
                probs = tf.nn.softmax(tf.squeeze(tf.stack(probs)));
                var action_inds = tf.stack([tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis:1);

                total_loss = np.zeros((NUM_TRAIN_EPOCHS));
                var act_loss = np.zeros((NUM_TRAIN_EPOCHS));
                var c_loss = np.zeros(((NUM_TRAIN_EPOCHS)));
                var ent_loss = np.zeros((NUM_TRAIN_EPOCHS));
                for (var epoch in range(NUM_TRAIN_EPOCHS))
                {
                    var loss_tuple = train_model(action_inds, tf.gather_nd(probs, action_inds),
                                             states, advantages, discounted_rewards, optimizer,
                                             ent_discount_val);
                    total_loss[epoch] = loss_tuple[0];
                    c_loss[epoch] = loss_tuple[1];
                    act_loss[epoch] = loss_tuple[2];
                    ent_loss[epoch] = loss_tuple[3];
                }
                var ent_discount_val *= ENT_DISCOUNT_RATE;

                using (train_writer.as_default())
                {
                    tf.summary.scalar("tot_loss", np.mean(total_loss), step);
                    tf.summary.scalar("critic_loss", np.mean(c_loss), step);
                    tf.summary.scalar("actor_loss", np.mean(act_loss), step);
                    tf.summary.scalar("entropy_loss", np.mean(ent_loss), step);
                }
            }
        }
    }
}