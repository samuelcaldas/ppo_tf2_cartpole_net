using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
//
using NumSharp;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
//import tensorflow as tf
//from tensorflow import keras
//import tensorflow_probability as tfp
//import numpy as np
//import gym
//import datetime as dt

namespace ppo_tf2_cartpole
{
    public class KerasModelArgs : ModelArgs
    {
        public int NumActions { get; set; }
    }
    public class KerasModel : Model
    { //(keras.Model)
        Layer dense1;
        Layer dense2;
        Layer value;
        Layer policy_logits;
        public KerasModel(KerasModelArgs args)
            : base(args)
        {
            //super().__init__();
            dense1 = keras.layers.Dense(64, activation: keras.activations.Relu,
                                        kernel_initializer: keras.initializers.he_normal());
            dense2 = keras.layers.Dense(64, activation: keras.activations.Relu,
                                        kernel_initializer: keras.initializers.he_normal());
            value = keras.layers.Dense(1);
            policy_logits = keras.layers.Dense(args.NumActions);
        }
        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            Tensor x = dense1.Apply(inputs);
            x = dense2.Apply(x);
            return (value.Apply(x), policy_logits.Apply(x));
        }
        public (Tensor, Tensor) action_value(Tensor state)
        {
            var (value, logits) = predict_on_batch(state);
            Tensor dist = tfp.distributions.Categorical(logits: logits);
            Tensor action = dist.sample();
            return (action, value);
        }
    }
    class Program
    {
        static string STORE_PATH = "C:\\Users\\andre\\TensorBoard\\PPOCartpole";
        static double CRITIC_LOSS_WEIGHT = 0.5;
        static double ENTROPY_LOSS_WEIGHT = 0.01;
        static double ENT_DISCOUNT_RATE = 0.995;
        static int BATCH_SIZE = 64;
        static double GAMMA = 0.99;
        static double CLIP_VALUE = 0.2;
        static double LR = 0.001;

        static int NUM_TRAIN_EPOCHS = 10;

        //static Tensor env = gym.make("CartPole-v0");
        static CartPoleEnv env = new CartPoleEnv(WinFormEnvViewer.Factory);
        static int state_size = 4;
        static int num_actions = env.ActionSpace.Shape.NDim;

        static double ent_discount_val = ENTROPY_LOSS_WEIGHT;

        static Model model = new KerasModel(new KerasModelArgs
        {
            NumActions = num_actions,
        });

        static Tensors critic_loss(Tensor discounted_rewards, Tensor value_est)
        {
            return tf.cast(tf.reduce_mean(keras.losses.MeanSquaredError().Call(discounted_rewards, value_est)) * CRITIC_LOSS_WEIGHT,
                           tf.float32);
        }
        static Tensors entropy_loss(Tensor policy_logits, Tensor ent_discount_val)
        {
            Tensor probs = tf.nn.softmax(policy_logits);
            Tensor entropy_loss = -tf.reduce_mean(keras.losses.CategoricalCrossentropy().Call(probs, probs));
            return entropy_loss * ent_discount_val;
        }
        static Tensors actor_loss(Tensor advantages, Tensor old_probs, Tensor action_inds, Tensor policy_logits)
        {
            Tensor probs = tf.nn.softmax(policy_logits);
            Tensor new_probs = tf.gather(probs, action_inds);

            Tensor ratio = new_probs / old_probs;

            Tensor policy_loss = -tf.reduce_mean(np.minimum(ratio * advantages,
                                                            tf.clip_by_value(ratio, 1.0 - CLIP_VALUE, 1.0 + CLIP_VALUE) * advantages));
            return policy_loss;
        }
        static (Tensor, Tensor, Tensor, Tensor) train_model(Tensor action_inds, Tensor old_probs, Tensor states, Tensor advantages, Tensor discounted_rewards, Tensor optimizer, Tensor ent_discount_val)
        {
            Tensor tot_loss;
            using (var tape = tf.GradientTape())
            {
                var modelpredict = model.Call(tf.stack(states));
                var (values, policy_logits) = (modelpredict[0], modelpredict[1]);
                Tensor act_loss = actor_loss(advantages, old_probs, action_inds, policy_logits);
                Tensor ent_loss = entropy_loss(policy_logits, ent_discount_val);
                Tensor c_loss = critic_loss(discounted_rewards, values);
                Tensor tot_loss = act_loss + ent_loss + c_loss;
            }
            Tensor grads = tape.gradient(tot_loss, model.trainable_variables);
            optimizer.apply_gradients(zip(grads, model.trainable_variables));
            return (tot_loss, c_loss, act_loss, ent_loss);
        }
        static (Tensor, Tensor) get_advantages(Tensor rewards, Tensor dones, NDArray values, Tensor next_value)
        {
            Tensor discounted_rewards = np.array(rewards + (next_value[0]));

            for (int t = 0; reversed(range(len(rewards))))
            {
                discounted_rewards[t] = rewards[t] + GAMMA * discounted_rewards[t + 1] * (1 - dones[t]);
            }
            discounted_rewards = discounted_rewards[":-1"];
            // advantages are bootstrapped discounted rewards - values, using Bellman's equation
            NDArray advantages = discounted_rewards - np.stack(values)[":, 0"];
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

            Tensor optimizer = keras.optimizers.Adam(learning_rate: LR);

            Tensor train_writer = tf.summary.create_file_writer(STORE_PATH + f"/PPO-CartPole_{dt.datetime.now().strftime(" % d % m % Y % H % M")}");

            int num_steps = 10000000;
            Double episode_reward_sum = 0;
            Tensor state = env.Reset();
            int episode = 1;
            Double total_loss = 0;
            for (int step = 0; step < num_steps; step++)
            {
                var rewards = new List<double>();
                var actions = new List<double>();
                var values = new List<double>();
                var states = new List<double>();
                var dones = new List<double>();
                var probs = new List<double>();
                for (int i = 0; i < BATCH_SIZE; i++)
                {
                    var (_, policy_logits) = model(state.reshape(1, -1));

                    var (action, value) = model.action_value(state.reshape(1, -1));
                    var (new_state, reward, done, _) = env.Step(action.numpy()[0]);

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
                        state = env.Reset();
                        if (total_loss != 0)
                        {
                            Console.WriteLine("Episode: {episode}, latest episode reward: {episode_reward_sum}, ",
                                              "total loss: {np.mean(total_loss)}, critic loss: {np.mean(c_loss)}, ",
                                              "actor loss: {np.mean(act_loss)}, entropy loss {np.mean(ent_loss)}");
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
                Tensor action_inds = tf.stack((tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)), axis: 1);

                total_loss = np.zeros((NUM_TRAIN_EPOCHS));
                Tensor act_loss = np.zeros((NUM_TRAIN_EPOCHS));
                Tensor c_loss = np.zeros(((NUM_TRAIN_EPOCHS)));
                Tensor ent_loss = np.zeros((NUM_TRAIN_EPOCHS));
                for (int epoch = 0; epoch < NUM_TRAIN_EPOCHS; epoch++)
                {
                    Tensor loss_tuple = train_model(action_inds, tf.gather(probs, action_inds),
                                             states, advantages, discounted_rewards, optimizer,
                                             ent_discount_val);
                    total_loss[epoch] = loss_tuple[0];
                    c_loss[epoch] = loss_tuple[1];
                    act_loss[epoch] = loss_tuple[2];
                    ent_loss[epoch] = loss_tuple[3];
                }
                Tensor ent_discount_val *= ENT_DISCOUNT_RATE;

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