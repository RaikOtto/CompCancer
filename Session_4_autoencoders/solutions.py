##########################
### Proposed solutions ###
##########################

##################
### Exercise 1 ###
##################

mrna_input = keras.layers.Input(shape=(input_size,), name="input")
hidden = keras.layers.Dense(embedding_size, activation="sigmoid", kernel_regularizer=keras.regularizers.l1(.001), name="hidden")(mrna_input)
output = keras.layers.Dense(input_size, activation="sigmoid", kernel_regularizer=keras.regularizers.l1(.001), name="reconstruction")(hidden)
l1_ae = tf.keras.Model(mrna_input, output, name="L1-regularized autoencoder")

l1_ae.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.mean_squared_error)
l1_ae.summary()

l1_ae.fit(
    x=x_train,
    y=x_train,
    validation_data=[x_test, x_test],
    epochs=1000,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3
    )]
)

pd.DataFrame(l1_ae.history.history).plot()
plt.ylim(0,2)

sns.distplot(l1_ae.layers[1].get_weights()[0].flatten(), kde=True)
plt.xlabel(r"$\|w\|$")

# this shows the distribution of weights to have a marked spike at 0, creating a sparser autoencoder.


##################
### Exercise 2 ###
##################
mrna_input = tf.keras.layers.Input(shape=(input_size,), name="input")
image_noisy = tf.keras.layers.GaussianNoise(stddev=noise_factor, name="noisy")(mrna_input)
hidden = tf.keras.layers.Dense(intermediate_size, activation="sigmoid", name="hidden")(image_noisy)
output = tf.keras.layers.Dense(input_size, activation="sigmoid", name="reconstruction")(hidden)
dae = tf.keras.Model(mrna_input, output)

dae.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.mean_squared_error)


##################
### Exercise 3 ###
##################

# Define the sampling function for the reparametrization trick
def sample(arg):
    z_mean, z_log_var = arg
    eps = tf.random.normal(shape=(z_mean.shape[1],))
    return z_mean + tf.multiply(tf.exp(z_log_var * 0.5), eps)

# Define the layers
mrna_input = tf.keras.layers.Input(shape=(input_size,), name="input")
h = tf.keras.layers.Dense(intermediate_size, activation="relu", name="hidden")(mrna_input)

# For the latent space, we learn the mean, and logvar
z_mean = tf.keras.layers.Dense(intermediate_size, activation="sigmoid", name="z_mean")(h)
z_log_var = tf.keras.layers.Dense(intermediate_size, name="z_log_var")(h)

# The latent factors are then obtained by sampling
z = tf.keras.layers.Lambda(sample, output_shape=(intermediate_size,), name="sample")([z_mean, z_log_var])

# The output layer is the same
output = tf.keras.layers.Dense(input_size, activation="sigmoid", name="reconstruction")(z)
vae = tf.keras.Model(mrna_input, output)


# We'll need to define the KL loss and add it to the vae
kl_loss = -0.5 * tf.reduce_sum(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
    axis=-1,
)
vae_loss = tf.reduce_mean(kl_loss)

# This adds the loss to the VAE
vae.add_loss(vae_loss)

# And we add the MSE loss when compiling
vae.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.mean_squared_error)

###################
### Exercise 3A ###
###################

deterministic_encoder = tf.keras.Model(mrna_input, z_mean)

